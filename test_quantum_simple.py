#!/usr/bin/env python3
"""
Simple test for quantum optimization without external dependencies
"""

import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_quantum_imports():
    """Test that quantum modules can be imported"""
    try:
        from sql_synthesizer.quantum.core import (
            QuantumQueryOptimizer, QuantumQueryPlanGenerator, 
            QueryPlan, QuantumState, Qubit
        )
        print("✅ Quantum core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Quantum core import failed: {e}")
        return False

def test_qubit_basic():
    """Test basic qubit functionality"""
    try:
        from sql_synthesizer.quantum.core import Qubit, QuantumState
        
        qubit = Qubit()
        print(f"  Qubit initialized: measured={qubit.measured}")
        
        value = qubit.measure()
        print(f"  Qubit measured: value={value}, measured={qubit.measured}")
        
        qubit.reset()
        print(f"  Qubit reset: measured={qubit.measured}")
        
        print("✅ Qubit basic test passed")
        return True
    except Exception as e:
        print(f"❌ Qubit basic test failed: {e}")
        return False

def test_query_plan():
    """Test query plan creation"""
    try:
        from sql_synthesizer.quantum.core import QueryPlan, QuantumState
        
        plan = QueryPlan(
            joins=[("users", "orders")],
            filters=[{"column": "id", "value": 1}],
            aggregations=["count"],
            cost=10.5,
            probability=0.8
        )
        
        print(f"  Plan created: cost={plan.cost}, state={plan.quantum_state}")
        print("✅ Query plan test passed")
        return True
    except Exception as e:
        print(f"❌ Query plan test failed: {e}")
        return False

def test_quantum_optimizer():
    """Test quantum optimizer basic functionality"""
    try:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer, QueryPlan
        
        optimizer = QuantumQueryOptimizer(num_qubits=4, temperature=100.0)
        print(f"  Optimizer created: qubits={len(optimizer.qubits)}, temp={optimizer.temperature}")
        
        # Test superposition
        plans = [
            QueryPlan([], [], [], 10.0, 0.0),
            QueryPlan([], [], [], 20.0, 0.0)
        ]
        
        superposition_plans = optimizer.create_superposition(plans)
        print(f"  Superposition created: {len(superposition_plans)} plans")
        
        # Test interference
        interfered_plans = optimizer.quantum_interference(superposition_plans)
        print(f"  Interference applied: plan probabilities updated")
        
        # Test annealing
        optimal_plan = optimizer.quantum_annealing(interfered_plans, iterations=10)
        print(f"  Annealing completed: optimal cost={optimal_plan.cost}")
        
        print("✅ Quantum optimizer test passed")
        return True
    except Exception as e:
        print(f"❌ Quantum optimizer test failed: {e}")
        return False

async def test_async_optimization():
    """Test async quantum optimization"""
    try:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer, QueryPlan
        
        optimizer = QuantumQueryOptimizer(num_qubits=4)
        plans = [
            QueryPlan([], [], [], 15.0, 0.0),
            QueryPlan([], [], [], 8.0, 0.0),
            QueryPlan([], [], [], 25.0, 0.0)
        ]
        
        optimal_plan = await optimizer.optimize_query_async(plans)
        print(f"  Async optimization completed: cost={optimal_plan.cost}")
        
        print("✅ Async optimization test passed")
        return True
    except Exception as e:
        print(f"❌ Async optimization test failed: {e}")
        return False

def test_plan_generator():
    """Test quantum plan generator"""
    try:
        from sql_synthesizer.quantum.core import QuantumQueryPlanGenerator
        
        generator = QuantumQueryPlanGenerator()
        
        plans = generator.generate_plans(
            tables=["users", "orders"],
            joins=[("users", "orders")],
            filters=[{"column": "status", "selectivity": 0.1}],
            aggregations=["count"]
        )
        
        print(f"  Generated {len(plans)} plans")
        if plans:
            print(f"  First plan cost: {plans[0].cost}")
        
        print("✅ Plan generator test passed")
        return True
    except Exception as e:
        print(f"❌ Plan generator test failed: {e}")
        return False

def test_scheduler_imports():
    """Test scheduler imports"""
    try:
        from sql_synthesizer.quantum.scheduler import (
            QuantumTaskScheduler, QuantumTask, TaskPriority, QuantumResource
        )
        print("✅ Scheduler imports successful")
        return True
    except ImportError as e:
        print(f"❌ Scheduler import failed: {e}")
        return False

def test_integration_imports():
    """Test integration imports"""
    try:
        from sql_synthesizer.quantum.integration import QuantumSQLSynthesizer
        print("✅ Integration imports successful")
        return True
    except ImportError as e:
        print(f"❌ Integration import failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing Quantum-Inspired SQL Optimization")
    print("=" * 50)
    
    tests = [
        ("Quantum Core Imports", test_quantum_imports),
        ("Qubit Basic Functionality", test_qubit_basic),
        ("Query Plan Creation", test_query_plan),
        ("Quantum Optimizer", test_quantum_optimizer),
        ("Plan Generator", test_plan_generator),
        ("Scheduler Imports", test_scheduler_imports),
        ("Integration Imports", test_integration_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Run async test
    print(f"\n🔬 Running: Async Optimization")
    try:
        if await test_async_optimization():
            passed += 1
        total += 1
    except Exception as e:
        print(f"❌ Async optimization failed with exception: {e}")
        total += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quantum tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))