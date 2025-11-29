"""
Main simulation module for Hospital Patient Flow Model.

Implements a process-based simulation using SimPy where each patient
is an independent process that flows through preparation, surgery, and recovery stages.
"""

import simpy
from typing import Callable
from config import SimulationConfig
from monitoring import SystemMonitor, monitoring_process
import random


class Patient:
    """Represents a patient in the system with personal service times."""
    
    def __init__(self, patient_id: int, prep_time: float, surgery_time: float,
                 recovery_time: float, patient_type: str, priority: int):
        self.patient_id = patient_id
        self.prep_time = prep_time
        self.surgery_time = surgery_time
        self.recovery_time = recovery_time
        self.patient_type = patient_type
        self.priority = priority
        self.arrival_time = None
        self.departure_time = None


def patient_process(env: simpy.Environment, patient: Patient,
                   prep_resource: simpy.Resource,
                   surgery_resource: simpy.Resource,
                   recovery_resource: simpy.Resource,
                   monitor: SystemMonitor):
    """Patient process flows through prep, surgery, and recovery."""
    patient.arrival_time = env.now
    monitor.record_patient_arrival(patient.patient_id, patient.arrival_time, patient.patient_type)
    
    with prep_resource.request() as prep_request:
        yield prep_request
        yield env.timeout(patient.prep_time)
    
    surgery_request = surgery_resource.request(priority=patient.priority)
    yield surgery_request
    yield env.timeout(patient.surgery_time)
    
    recovery_request = recovery_resource.request()
    was_blocked = recovery_resource.count >= recovery_resource.capacity
    if was_blocked:
        monitor.start_blocking(env.now)
    
    yield recovery_request
    
    if was_blocked:
        monitor.end_blocking(env.now)
    
    surgery_resource.release(surgery_request)
    yield env.timeout(patient.recovery_time)
    recovery_resource.release(recovery_request)
    
    patient.departure_time = env.now
    monitor.record_patient_departure(patient.patient_id, patient.departure_time)


def arrival_generator(env: simpy.Environment,
                     prep_resource: simpy.Resource,
                     surgery_resource: simpy.Resource,
                     recovery_resource: simpy.Resource,
                     monitor: SystemMonitor,
                     interarrival_gen: Callable[[], float],
                     prep_time_gen: Callable[[], float],
                     surgery_time_gen: Callable[[], float],
                     recovery_time_gen: Callable[[], float]):
    """Generates patient arrivals with personal service times."""
    patient_id = 0
    
    while True:
        interarrival_time = interarrival_gen()
        yield env.timeout(interarrival_time)
        
        prep_time = prep_time_gen()
        surgery_time = surgery_time_gen()
        recovery_time = recovery_time_gen()

        if random.random() < 0.1:
            patient_type = "urgent"
            priority = 1
        else:
            patient_type = "routine"
            priority = 2
        
        patient = Patient(
            patient_id,
            prep_time,
            surgery_time,
            recovery_time,
            patient_type,
            priority
        )

        
        env.process(patient_process(
            env, patient,
            prep_resource, surgery_resource, recovery_resource,
            monitor
        ))
        
        patient_id += 1


def run_simulation(config: SimulationConfig) -> SystemMonitor:
    """Run the simulation and return monitor with statistics."""
    config.initialize_random_seed()
    
    env = simpy.Environment()
    
    prep_resource = simpy.Resource(env, capacity=config.num_prep_rooms)
    surgery_resource = simpy.PriorityResource(env, capacity=config.num_operating_theatres)
    recovery_resource = simpy.Resource(env, capacity=config.num_recovery_rooms)
    
    monitor = SystemMonitor(env, config.monitoring_interval)
    
    env.process(arrival_generator(
        env,
        prep_resource, surgery_resource, recovery_resource,
        monitor,
        config.get_interarrival_generator(),
        config.get_prep_time_generator(),
        config.get_surgery_time_generator(),
        config.get_recovery_time_generator()
    ))
    
    env.process(monitoring_process(
        env, monitor,
        prep_resource, surgery_resource, recovery_resource,
        config.monitoring_interval
    ))
    
    env.run(until=config.simulation_time)
    
    monitor.finalize(env.now, prep_resource, surgery_resource, recovery_resource)
    
    return monitor


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    print("Starting Hospital Patient Flow Simulation...")
    print(f"Configuration:")
    print(f"  Preparation rooms: {DEFAULT_CONFIG.num_prep_rooms}")
    print(f"  Recovery rooms: {DEFAULT_CONFIG.num_recovery_rooms}")
    print(f"  Inter-arrival mean: {DEFAULT_CONFIG.interarrival_mean}")
    print(f"  Prep time mean: {DEFAULT_CONFIG.prep_time_mean}")
    print(f"  Surgery time mean: {DEFAULT_CONFIG.surgery_time_mean}")
    print(f"  Recovery time mean: {DEFAULT_CONFIG.recovery_time_mean}")
    print(f"  Simulation time: {DEFAULT_CONFIG.simulation_time}")
    print()
    
    monitor = run_simulation(DEFAULT_CONFIG)
    
    report = monitor.generate_report()
    print(report)
    
    with open("simulation_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to simulation_report.txt")

