from dataclasses import dataclass
from typing import Callable
import random


@dataclass
class SimulationConfig:
    num_prep_rooms: int = 3
    num_operating_theatres: int = 1
    num_recovery_rooms: int = 3
    interarrival_mean: float = 25.0
    prep_time_mean: float = 40.0
    surgery_time_mean: float = 20.0
    recovery_time_mean: float = 40.0
    simulation_time: float = 10000.0
    monitoring_interval: float = 1.0
    random_seed: int = None
    
    interarrival_dist: str = "exponential"
    interarrival_param2: float = None
    
    prep_dist: str = "exponential"
    prep_param2: float = None
    
    recovery_dist: str = "exponential"
    recovery_param2: float = None
    
    def get_interarrival_generator(self) -> Callable[[], float]:
        if self.interarrival_dist == "uniform":
            min_val = self.interarrival_mean
            max_val = self.interarrival_param2
            def generator():
                return random.uniform(min_val, max_val)
            return generator
        else:
            def generator():
                return random.expovariate(1.0 / self.interarrival_mean)
            return generator
    
    def get_prep_time_generator(self) -> Callable[[], float]:
        if self.prep_dist == "uniform":
            min_val = self.prep_time_mean
            max_val = self.prep_param2 if self.prep_param2 else 50.0
            def generator():
                return random.uniform(min_val, max_val)
            return generator
        else:
            def generator():
                return random.expovariate(1.0 / self.prep_time_mean)
            return generator
    
    def get_surgery_time_generator(self) -> Callable[[], float]:
        def generator():
            return random.expovariate(1.0 / self.surgery_time_mean)
        return generator
    
    def get_recovery_time_generator(self) -> Callable[[], float]:
        if self.recovery_dist == "uniform":
            min_val = self.recovery_time_mean
            max_val = self.recovery_param2 if self.recovery_param2 else 50.0
            def generator():
                return random.uniform(min_val, max_val)
            return generator
        else:
            def generator():
                return random.expovariate(1.0 / self.recovery_time_mean)
            return generator
    
    def initialize_random_seed(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)


DEFAULT_CONFIG = SimulationConfig(
    num_prep_rooms=3,
    num_recovery_rooms=3,
    interarrival_mean=25.0,
    prep_time_mean=40.0,
    surgery_time_mean=20.0,
    recovery_time_mean=40.0,
    simulation_time=10000.0,
    monitoring_interval=1.0,
    random_seed=42
)

