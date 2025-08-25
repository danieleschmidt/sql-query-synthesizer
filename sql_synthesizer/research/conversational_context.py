"""Conversational Context Management for NL2SQL.

This module implements a novel conversational context management system that maintains
semantic representations of previous queries and enables more accurate SQL synthesis
for follow-up queries.

Research Hypothesis: Maintaining conversation context and query history enables more
accurate and natural SQL synthesis for follow-up queries.

Expected Impact: 30-50% improvement in conversational query accuracy.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using fallback implementation.")


@dataclass
class QueryContext:
    """Represents context for a single query in conversation."""
    query_id: str
    timestamp: float
    natural_language: str
    generated_sql: str
    execution_result: Optional[Dict[str, Any]]
    entities_mentioned: List[str]
    operations_performed: List[str]
    tables_accessed: List[str]
    columns_referenced: List[str]
    success: bool
    confidence: float
    user_feedback: Optional[str] = None
    semantic_embedding: Optional[List[float]] = None


@dataclass
class ConversationSession:
    """Represents an entire conversation session."""
    session_id: str
    start_time: float
    last_activity: float
    queries: List[QueryContext]
    session_metadata: Dict[str, Any]
    global_context: Dict[str, Any]  # Persistent context across queries
    

class SemanticEntityTracker:
    """Tracks entities and their references throughout conversation."""
    
    def __init__(self):
        self.entity_aliases = defaultdict(set)  # Maps canonical name to aliases
        self.entity_context = defaultdict(dict)  # Stores context for each entity
        self.pronoun_resolution = {}  # Maps pronouns to entities
        
    def register_entity(self, canonical_name: str, aliases: List[str], 
                       context: Dict[str, Any]):
        """Register an entity with its aliases and context."""
        self.entity_aliases[canonical_name].update(aliases)
        self.entity_aliases[canonical_name].add(canonical_name)
        self.entity_context[canonical_name].update(context)
        
    def resolve_reference(self, reference: str, conversation_context: List[str]) -> Optional[str]:
        """Resolve a reference to its canonical entity name."""
        # Direct match
        for canonical, aliases in self.entity_aliases.items():
            if reference.lower() in [alias.lower() for alias in aliases]:
                return canonical
                
        # Pronoun resolution
        pronouns = ['it', 'they', 'them', 'that', 'those', 'this', 'these']
        if reference.lower() in pronouns:
            # Find most recently mentioned entity
            for query in reversed(conversation_context):
                for canonical, aliases in self.entity_aliases.items():
                    if any(alias.lower() in query.lower() for alias in aliases):
                        self.pronoun_resolution[reference.lower()] = canonical
                        return canonical
        
        # Partial match
        for canonical, aliases in self.entity_aliases.items():
            for alias in aliases:
                if reference.lower() in alias.lower() or alias.lower() in reference.lower():
                    return canonical
                    
        return None
        
    def get_entity_context(self, entity: str) -> Dict[str, Any]:
        """Get stored context for an entity."""
        return self.entity_context.get(entity, {})
        
    def update_entity_context(self, entity: str, new_context: Dict[str, Any]):
        """Update context for an entity."""
        self.entity_context[entity].update(new_context)


class IntentDisambiguator:
    """Disambiguates query intent based on conversation history."""
    
    def __init__(self):
        self.intent_patterns = {
            'comparison': ['compare', 'vs', 'versus', 'difference', 'better', 'worse'],
            'continuation': ['also', 'and', 'additionally', 'furthermore', 'what about'],
            'refinement': ['but', 'however', 'instead', 'actually', 'rather'],
            'drill_down': ['details', 'breakdown', 'specifically', 'more info', 'elaborate'],
            'aggregation': ['total', 'sum', 'count', 'average', 'maximum', 'minimum'],
            'temporal': ['when', 'during', 'before', 'after', 'since', 'until'],
            'filter': ['only', 'just', 'excluding', 'without', 'filter', 'where']
        }
        
    def classify_intent(self, current_query: str, previous_queries: List[str]) -> Dict[str, float]:
        """Classify the intent of current query in context of previous queries."""
        intent_scores = defaultdict(float)
        current_lower = current_query.lower()
        
        # Pattern-based classification
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in current_lower:
                    intent_scores[intent] += 1.0
                    
        # Context-based classification
        if previous_queries:
            last_query = previous_queries[-1].lower()
            
            # Check for continuation patterns
            continuation_words = ['also', 'and', 'what about', 'how about']
            if any(word in current_lower for word in continuation_words):
                intent_scores['continuation'] += 2.0
                
            # Check for refinement patterns
            refinement_words = ['but', 'however', 'actually', 'instead']
            if any(word in current_lower for word in refinement_words):
                intent_scores['refinement'] += 2.0
                
            # Check for comparison if previous query mentioned entities
            if ('compare' in current_lower or 'vs' in current_lower) and len(previous_queries) > 0:
                intent_scores['comparison'] += 2.0
                
        # Normalize scores
        total_score = sum(intent_scores.values()) or 1.0
        return {intent: score / total_score for intent, score in intent_scores.items()}
        
    def resolve_ambiguous_references(self, query: str, entity_tracker: SemanticEntityTracker,
                                   conversation_history: List[str]) -> str:
        """Resolve ambiguous references in the query."""
        resolved_query = query
        
        # Find potential references to resolve
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['it', 'they', 'them', 'that', 'those', 'this', 'these']:
                resolved_entity = entity_tracker.resolve_reference(word, conversation_history)
                if resolved_entity:
                    words[i] = resolved_entity
                    
        return ' '.join(words)


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history_length: int = 50, session_timeout_hours: int = 24):
        self.sessions = {}  # session_id -> ConversationSession
        self.max_history_length = max_history_length
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.entity_tracker = SemanticEntityTracker()
        self.intent_disambiguator = IntentDisambiguator()
        
        # Global statistics
        self.global_stats = {
            'total_sessions': 0,
            'total_queries': 0,
            'average_session_length': 0.0,
            'common_entities': defaultdict(int),
            'intent_distribution': defaultdict(int)
        }
        
    def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            session_id=session_id,
            start_time=time.time(),
            last_activity=time.time(),
            queries=[],
            session_metadata=metadata or {},
            global_context={}
        )
        self.sessions[session_id] = session
        self.global_stats['total_sessions'] += 1
        return session
        
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get existing session or None if not found/expired."""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # Check if session has expired
        if (datetime.now() - datetime.fromtimestamp(session.last_activity)) > self.session_timeout:
            del self.sessions[session_id]
            return None
            
        return session
        
    def add_query_context(self, session_id: str, query_context: QueryContext):
        """Add a query to the conversation history."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
            
        session.queries.append(query_context)
        session.last_activity = time.time()
        
        # Maintain history length limit
        if len(session.queries) > self.max_history_length:
            session.queries = session.queries[-self.max_history_length:]
            
        # Update entity tracker
        for entity in query_context.entities_mentioned:
            self.entity_tracker.register_entity(
                entity, 
                [entity], 
                {'last_mentioned': query_context.timestamp, 'context': 'query'}
            )
            self.global_stats['common_entities'][entity] += 1
            
        # Update global statistics
        self.global_stats['total_queries'] += 1
        self._update_session_stats()
        
    def get_context_for_query(self, session_id: str, lookback_queries: int = 5) -> Dict[str, Any]:
        """Get relevant context for processing a new query."""
        session = self.get_session(session_id)
        if not session:
            return {'queries': [], 'entities': {}, 'global_context': {}}
            
        recent_queries = session.queries[-lookback_queries:] if session.queries else []
        
        # Collect entities from recent queries
        recent_entities = set()
        for query in recent_queries:
            recent_entities.update(query.entities_mentioned)
            
        # Get entity context
        entity_contexts = {
            entity: self.entity_tracker.get_entity_context(entity) 
            for entity in recent_entities
        }
        
        # Collect recent tables and columns
        recent_tables = set()
        recent_columns = set()
        for query in recent_queries:
            recent_tables.update(query.tables_accessed)
            recent_columns.update(query.columns_referenced)
            
        return {
            'queries': [
                {
                    'natural_language': q.natural_language,
                    'sql': q.generated_sql,
                    'timestamp': q.timestamp,
                    'success': q.success,
                    'confidence': q.confidence
                } for q in recent_queries
            ],
            'entities': entity_contexts,
            'recent_tables': list(recent_tables),
            'recent_columns': list(recent_columns),
            'global_context': session.global_context,
            'session_metadata': session.session_metadata
        }
        
    def resolve_query_references(self, session_id: str, current_query: str) -> str:
        """Resolve ambiguous references in current query using conversation history."""
        session = self.get_session(session_id)
        if not session:
            return current_query
            
        # Get conversation history as strings
        history = [q.natural_language for q in session.queries]
        
        # Resolve references
        resolved_query = self.intent_disambiguator.resolve_ambiguous_references(
            current_query, self.entity_tracker, history
        )
        
        return resolved_query
        
    def classify_query_intent(self, session_id: str, current_query: str) -> Dict[str, float]:
        """Classify the intent of the current query in conversation context."""
        session = self.get_session(session_id)
        if not session:
            return {'standalone': 1.0}
            
        history = [q.natural_language for q in session.queries]
        intent_scores = self.intent_disambiguator.classify_intent(current_query, history)
        
        # Update global intent statistics
        for intent, score in intent_scores.items():
            self.global_stats['intent_distribution'][intent] += score
            
        return intent_scores
        
    def update_query_feedback(self, session_id: str, query_id: str, feedback: str, success: bool):
        """Update feedback for a specific query."""
        session = self.get_session(session_id)
        if not session:
            return
            
        for query in session.queries:
            if query.query_id == query_id:
                query.user_feedback = feedback
                query.success = success
                break
                
    def _update_session_stats(self):
        """Update global session statistics."""
        if self.sessions:
            total_queries = sum(len(session.queries) for session in self.sessions.values())
            self.global_stats['average_session_length'] = total_queries / len(self.sessions)
            
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global conversation statistics."""
        return dict(self.global_stats)
        
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - datetime.fromtimestamp(session.last_activity)) > self.session_timeout:
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
        return len(expired_sessions)


if TRANSFORMERS_AVAILABLE:
    class ContextualEncoder(nn.Module):
        """Neural encoder for conversation context using transformers."""
        
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            super().__init__()
            self.model_name = model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Could not load transformer model {model_name}: {e}")
                self.tokenizer = None
                self.encoder = None
                
            # Context combination layers
            if self.encoder:
                hidden_size = self.encoder.config.hidden_size
                self.context_combiner = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                
        def encode_query(self, query: str) -> torch.Tensor:
            """Encode a single query into a semantic embedding."""
            if not self.encoder or not self.tokenizer:
                # Fallback: simple hash-based encoding
                return torch.tensor([hash(query) % 1000 / 1000.0] * 384, dtype=torch.float32)
                
            with torch.no_grad():
                inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
                outputs = self.encoder(**inputs)
                # Use CLS token or mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding.squeeze()
                
        def encode_conversation_context(self, queries: List[str], weights: Optional[List[float]] = None) -> torch.Tensor:
            """Encode conversation context with temporal weighting."""
            if not queries:
                return torch.zeros(384 if self.encoder else 384, dtype=torch.float32)
                
            embeddings = []
            for query in queries:
                embedding = self.encode_query(query)
                embeddings.append(embedding)
                
            # Stack embeddings
            stacked = torch.stack(embeddings)
            
            # Apply temporal weighting (recent queries get higher weight)
            if weights is None:
                weights = [0.5 ** (len(queries) - i - 1) for i in range(len(queries))]
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            
            # Weighted average
            weighted_context = (stacked * weights).sum(dim=0) / weights.sum()
            
            return weighted_context
            
        def combine_query_and_context(self, current_query: str, context_queries: List[str]) -> torch.Tensor:
            """Combine current query with conversation context."""
            current_embedding = self.encode_query(current_query)
            context_embedding = self.encode_conversation_context(context_queries)
            
            if self.encoder and hasattr(self, 'context_combiner'):
                combined = torch.cat([current_embedding, context_embedding], dim=-1)
                return self.context_combiner(combined)
            else:
                # Simple average fallback
                return (current_embedding + context_embedding) / 2

else:
    class ContextualEncoder:
        """Fallback encoder without transformers."""
        
        def __init__(self, model_name: str = "fallback"):
            self.model_name = model_name
            logger.warning("Using fallback contextual encoder without transformers")
            
        def encode_query(self, query: str) -> List[float]:
            """Simple hash-based encoding."""
            # Create a simple feature vector based on query characteristics
            features = []
            words = query.lower().split()
            
            # Length features
            features.append(len(words) / 20.0)  # Normalized word count
            features.append(len(query) / 100.0)  # Normalized character count
            
            # Keyword features
            sql_keywords = ['select', 'from', 'where', 'join', 'group', 'order', 'count', 'sum', 'avg']
            for keyword in sql_keywords:
                features.append(1.0 if keyword in query.lower() else 0.0)
                
            # Question word features
            question_words = ['what', 'how', 'when', 'where', 'which', 'who', 'why']
            for qword in question_words:
                features.append(1.0 if qword in query.lower() else 0.0)
                
            # Pad or truncate to fixed size
            target_size = 32
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
                
            return features
            
        def encode_conversation_context(self, queries: List[str], weights: Optional[List[float]] = None) -> List[float]:
            """Encode conversation context."""
            if not queries:
                return [0.0] * 32
                
            embeddings = [self.encode_query(q) for q in queries]
            
            # Simple weighted average
            if weights is None:
                weights = [0.5 ** (len(queries) - i - 1) for i in range(len(queries))]
                
            weighted_sum = [0.0] * len(embeddings[0])
            weight_sum = sum(weights)
            
            for embedding, weight in zip(embeddings, weights):
                for i, val in enumerate(embedding):
                    weighted_sum[i] += val * weight
                    
            return [val / weight_sum for val in weighted_sum]
            
        def combine_query_and_context(self, current_query: str, context_queries: List[str]) -> List[float]:
            """Combine current query with context."""
            current_embedding = self.encode_query(current_query)
            context_embedding = self.encode_conversation_context(context_queries)
            
            # Simple average
            combined = []
            for curr, ctx in zip(current_embedding, context_embedding):
                combined.append((curr + ctx) / 2.0)
                
            return combined


class ConversationalNL2SQL:
    """Main conversational NL2SQL system with context awareness."""
    
    def __init__(self, 
                 base_sql_generator: Optional[Any] = None,
                 encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_context_queries: int = 5):
        
        self.base_sql_generator = base_sql_generator
        self.conversation_memory = ConversationMemory()
        self.contextual_encoder = ContextualEncoder(encoder_model)
        self.max_context_queries = max_context_queries
        
        # Performance tracking
        self.performance_metrics = {
            'contextual_queries': 0,
            'standalone_queries': 0,
            'resolution_success_rate': 0.0,
            'average_context_relevance': 0.0,
            'intent_classification_accuracy': 0.0
        }
        
    def process_conversational_query(self, 
                                   natural_language: str,
                                   schema_metadata: Dict[str, Any],
                                   session_id: str,
                                   query_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query with conversational context awareness."""
        
        query_id = query_id or f"query_{int(time.time() * 1000)}"
        
        try:
            # Get conversation context
            context = self.conversation_memory.get_context_for_query(
                session_id, self.max_context_queries
            )
            
            # Resolve ambiguous references
            resolved_query = self.conversation_memory.resolve_query_references(
                session_id, natural_language
            )
            
            # Classify query intent
            intent_scores = self.conversation_memory.classify_query_intent(
                session_id, natural_language
            )
            
            # Determine if this is a contextual query
            is_contextual = max(intent_scores.values()) > 0.3 if intent_scores else False
            
            # Generate context-aware prompt if using base generator
            if self.base_sql_generator and hasattr(self.base_sql_generator, 'synthesize_sql'):
                if is_contextual and context['queries']:
                    # Enhance prompt with context
                    context_prompt = self._build_contextual_prompt(
                        resolved_query, context, intent_scores
                    )
                    result = self.base_sql_generator.synthesize_sql(context_prompt, schema_metadata)
                else:
                    # Use original query
                    result = self.base_sql_generator.synthesize_sql(resolved_query, schema_metadata)
            else:
                # Fallback SQL generation
                result = self._generate_fallback_sql(resolved_query, schema_metadata, context)
            
            # Extract entities and metadata
            entities = self._extract_entities_from_query(resolved_query)
            operations = self._extract_operations_from_sql(result.get('sql', ''))
            
            # Create query context
            query_context = QueryContext(
                query_id=query_id,
                timestamp=time.time(),
                natural_language=natural_language,
                generated_sql=result.get('sql', ''),
                execution_result=None,  # To be filled after execution
                entities_mentioned=entities,
                operations_performed=operations,
                tables_accessed=result.get('selected_tables', []),
                columns_referenced=result.get('selected_columns', []),
                success=True,  # Will be updated based on execution
                confidence=result.get('confidence', 0.5),
                semantic_embedding=self._get_query_embedding(resolved_query)
            )
            
            # Add to conversation memory
            self.conversation_memory.add_query_context(session_id, query_context)
            
            # Update performance metrics
            if is_contextual:
                self.performance_metrics['contextual_queries'] += 1
            else:
                self.performance_metrics['standalone_queries'] += 1
                
            # Enhance result with conversational metadata
            result.update({
                'query_id': query_id,
                'session_id': session_id,
                'resolved_query': resolved_query,
                'original_query': natural_language,
                'is_contextual': is_contextual,
                'intent_scores': intent_scores,
                'context_used': len(context['queries']),
                'entities_resolved': len(entities),
                'conversation_length': len(context['queries']) + 1,
                'processing_method': 'conversational_context_aware'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conversational processing: {str(e)}")
            return {
                'sql': 'SELECT 1 as error;',
                'explanation': f'Error in conversational processing: {str(e)}',
                'confidence': 0.0,
                'error': str(e),
                'query_id': query_id,
                'session_id': session_id
            }
            
    def _build_contextual_prompt(self, 
                               current_query: str, 
                               context: Dict[str, Any],
                               intent_scores: Dict[str, float]) -> str:
        """Build enhanced prompt with conversational context."""
        
        prompt_parts = []
        
        # Add conversation history context
        if context['queries']:
            prompt_parts.append("Previous conversation:")
            for i, prev_query in enumerate(context['queries'][-3:]):  # Last 3 queries
                prompt_parts.append(f"Q{i+1}: {prev_query['natural_language']}")
                prompt_parts.append(f"SQL{i+1}: {prev_query['sql']}")
                
        # Add entity context
        if context['entities']:
            entities_str = ', '.join(context['entities'].keys())
            prompt_parts.append(f"Entities in context: {entities_str}")
            
        # Add recent tables/columns
        if context['recent_tables']:
            tables_str = ', '.join(context['recent_tables'])
            prompt_parts.append(f"Recently used tables: {tables_str}")
            
        # Add intent information
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'standalone'
        prompt_parts.append(f"Query intent: {primary_intent}")
        
        # Add current query
        prompt_parts.append(f"Current question: {current_query}")
        
        return "\n".join(prompt_parts)
        
    def _generate_fallback_sql(self, 
                             query: str, 
                             schema_metadata: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback SQL when no base generator is available."""
        
        # Simple rule-based SQL generation
        query_lower = query.lower()
        
        # Find tables mentioned in query or context
        all_tables = list(schema_metadata.get('tables', {}).keys())
        mentioned_tables = []
        
        for table in all_tables:
            if table.lower() in query_lower:
                mentioned_tables.append(table)
                
        # Use context tables if none found
        if not mentioned_tables and context.get('recent_tables'):
            mentioned_tables = context['recent_tables'][:2]  # Use up to 2 recent tables
            
        # Fallback to first table
        if not mentioned_tables and all_tables:
            mentioned_tables = [all_tables[0]]
            
        # Determine operation type
        if any(word in query_lower for word in ['count', 'how many', 'number']):
            sql = f"SELECT COUNT(*) FROM {mentioned_tables[0] if mentioned_tables else 'table_name'}"
        elif any(word in query_lower for word in ['sum', 'total']):
            sql = f"SELECT SUM(column_name) FROM {mentioned_tables[0] if mentioned_tables else 'table_name'}"
        elif any(word in query_lower for word in ['average', 'avg', 'mean']):
            sql = f"SELECT AVG(column_name) FROM {mentioned_tables[0] if mentioned_tables else 'table_name'}"
        else:
            columns = "*"
            if mentioned_tables and mentioned_tables[0] in schema_metadata.get('tables', {}):
                table_info = schema_metadata['tables'][mentioned_tables[0]]
                if 'columns' in table_info:
                    column_names = [col['name'] for col in table_info['columns'][:3]]
                    columns = ', '.join(column_names)
                    
            sql = f"SELECT {columns} FROM {mentioned_tables[0] if mentioned_tables else 'table_name'} LIMIT 10"
            
        return {
            'sql': sql,
            'explanation': 'Generated using conversational fallback with context',
            'confidence': 0.6,
            'selected_tables': mentioned_tables,
            'selected_columns': []
        }
        
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity mentions from query."""
        entities = []
        words = query.lower().split()
        
        # Common entity patterns
        entity_patterns = [
            'customer', 'user', 'account', 'order', 'product', 'item',
            'payment', 'transaction', 'invoice', 'employee', 'department',
            'category', 'brand', 'supplier', 'vendor'
        ]
        
        for word in words:
            if word in entity_patterns:
                entities.append(word)
                
        return list(set(entities))  # Remove duplicates
        
    def _extract_operations_from_sql(self, sql: str) -> List[str]:
        """Extract SQL operations from generated SQL."""
        operations = []
        sql_upper = sql.upper()
        
        operation_patterns = {
            'SELECT': 'select',
            'COUNT': 'count',
            'SUM': 'sum',
            'AVG': 'average',
            'MAX': 'maximum',
            'MIN': 'minimum',
            'JOIN': 'join',
            'GROUP BY': 'group',
            'ORDER BY': 'order',
            'WHERE': 'filter'
        }
        
        for pattern, operation in operation_patterns.items():
            if pattern in sql_upper:
                operations.append(operation)
                
        return operations
        
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get semantic embedding for query."""
        try:
            if TRANSFORMERS_AVAILABLE and hasattr(self.contextual_encoder, 'encode_query'):
                embedding = self.contextual_encoder.encode_query(query)
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                elif isinstance(embedding, list):
                    return embedding
            else:
                return self.contextual_encoder.encode_query(query)
        except Exception as e:
            logger.warning(f"Could not generate embedding: {e}")
            return None
            
    def update_query_execution_result(self, 
                                    session_id: str, 
                                    query_id: str,
                                    execution_result: Dict[str, Any],
                                    success: bool):
        """Update query context with execution results."""
        session = self.conversation_memory.get_session(session_id)
        if not session:
            return
            
        for query_ctx in session.queries:
            if query_ctx.query_id == query_id:
                query_ctx.execution_result = execution_result
                query_ctx.success = success
                break
                
    def provide_user_feedback(self, 
                            session_id: str, 
                            query_id: str, 
                            feedback: str,
                            rating: Optional[float] = None):
        """Provide user feedback for a query."""
        self.conversation_memory.update_query_feedback(
            session_id, query_id, feedback, rating is not None and rating > 0.5
        )
        
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session."""
        session = self.conversation_memory.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
            
        successful_queries = [q for q in session.queries if q.success]
        failed_queries = [q for q in session.queries if not q.success]
        
        # Calculate statistics
        avg_confidence = np.mean([q.confidence for q in session.queries]) if session.queries else 0.0
        
        # Get most mentioned entities
        entity_counts = defaultdict(int)
        for query in session.queries:
            for entity in query.entities_mentioned:
                entity_counts[entity] += 1
                
        return {
            'session_id': session_id,
            'start_time': session.start_time,
            'duration_minutes': (session.last_activity - session.start_time) / 60,
            'total_queries': len(session.queries),
            'successful_queries': len(successful_queries),
            'failed_queries': len(failed_queries),
            'success_rate': len(successful_queries) / len(session.queries) if session.queries else 0,
            'average_confidence': avg_confidence,
            'most_mentioned_entities': dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'unique_tables_accessed': len(set(table for q in session.queries for table in q.tables_accessed)),
            'conversation_patterns': self._analyze_conversation_patterns(session.queries)
        }
        
    def _analyze_conversation_patterns(self, queries: List[QueryContext]) -> Dict[str, Any]:
        """Analyze patterns in the conversation."""
        if len(queries) < 2:
            return {'pattern': 'single_query'}
            
        # Check for drill-down pattern
        drill_down_score = 0
        for i in range(1, len(queries)):
            if (len(queries[i].tables_accessed) >= len(queries[i-1].tables_accessed) and
                len(queries[i].columns_referenced) > len(queries[i-1].columns_referenced)):
                drill_down_score += 1
                
        # Check for exploration pattern
        unique_tables = set()
        for query in queries:
            unique_tables.update(query.tables_accessed)
        exploration_score = len(unique_tables) / len(queries) if queries else 0
        
        # Check for refinement pattern
        refinement_score = 0
        for i in range(1, len(queries)):
            if (set(queries[i].tables_accessed) == set(queries[i-1].tables_accessed) and
                len(queries[i].operations_performed) != len(queries[i-1].operations_performed)):
                refinement_score += 1
                
        patterns = {
            'drill_down_score': drill_down_score / (len(queries) - 1),
            'exploration_score': exploration_score,
            'refinement_score': refinement_score / (len(queries) - 1)
        }
        
        # Determine primary pattern
        max_pattern = max(patterns.items(), key=lambda x: x[1])
        patterns['primary_pattern'] = max_pattern[0].replace('_score', '')
        
        return patterns
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'conversation_statistics': self.conversation_memory.get_global_statistics(),
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'encoder_model': self.contextual_encoder.model_name,
                'max_context_queries': self.max_context_queries
            },
            'active_sessions': len(self.conversation_memory.sessions),
            'total_entities_tracked': len(self.conversation_memory.entity_tracker.entity_aliases)
        }
        
    def cleanup_old_sessions(self) -> int:
        """Clean up expired sessions and return number cleaned."""
        return self.conversation_memory.cleanup_expired_sessions()


# Export main classes
__all__ = [
    'ConversationalNL2SQL',
    'ConversationMemory',
    'QueryContext',
    'SemanticEntityTracker',
    'IntentDisambiguator',
    'ContextualEncoder',
    'TRANSFORMERS_AVAILABLE'
]