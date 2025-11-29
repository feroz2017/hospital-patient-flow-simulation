import simpy
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class QueueStats:
    samples: List[int] = field(default_factory=list)
    total_waiting_time: float = 0.0
    max_length: int = 0
    
    def add_sample(self, length: int, duration: float):
        self.samples.append(length)
        self.total_waiting_time += length * duration
        if length > self.max_length:
            self.max_length = length
    
    def get_average_length(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
    
    def get_time_weighted_average(self, total_time: float) -> float:
        if total_time == 0:
            return 0.0
        return self.total_waiting_time / total_time


@dataclass
class ResourceUtilization:
    busy_time: float = 0.0
    total_time: float = 0.0
    capacity: int = 1
    
    def get_utilization(self) -> float:
        if self.total_time == 0 or self.capacity == 0:
            return 0.0
        return (self.busy_time / (self.total_time * self.capacity)) * 100.0


@dataclass
class PatientRecord:
    patient_id: int
    arrival_time: float
    patient_type: str
    departure_time: float = None
    
    def get_throughput_time(self) -> float:
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time


class SystemMonitor:
    def __init__(self, env: simpy.Environment, monitoring_interval: float = 1.0):
        self.env = env
        self.monitoring_interval = monitoring_interval
        
        self.prep_queue_stats = QueueStats()
        self.surgery_queue_stats = QueueStats()
        self.recovery_queue_stats = QueueStats()
        
        self.prep_utilization = ResourceUtilization()
        self.theatre_utilization = ResourceUtilization()
        self.recovery_utilization = ResourceUtilization()
        
        self.patient_records: Dict[int, PatientRecord] = {}
        
        self.blocking_events: List[tuple] = []
        self.current_blocking_start: float = None
        self.last_sample_time: float = 0.0

        self.time_all_recovery_busy = 0.0
        self.total_observed_time = 0.0

        self.urgent_throughputs = []
        self.routine_throughputs = []
    
    def record_patient_arrival(self, patient_id: int, arrival_time: float, patient_type: str):
        self.patient_records[patient_id] = PatientRecord(
            patient_id=patient_id,
            arrival_time=arrival_time,
            patient_type=patient_type
        )

    def get_recovery_full_probability(self):
    	if self.total_observed_time == 0:
        	return 0.0
    	return (self.time_all_recovery_busy / self.total_observed_time) * 100

    
    def record_patient_departure(self, patient_id: int, departure_time: float):
        if patient_id in self.patient_records:
            #self.patient_records[patient_id].departure_time = departure_time
            record = self.patient_records[patient_id]
            record.departure_time = departure_time

            throughput = record.departure_time - record.arrival_time

        if hasattr(record, 'patient_type'):
            if record.patient_type == "urgent":
                self.urgent_throughputs.append(throughput)
            else:
                self.routine_throughputs.append(throughput)
    
    def sample_queue_lengths(self, prep_queue_len: int, surgery_queue_len: int, 
                            recovery_queue_len: int, duration: float):
        self.prep_queue_stats.add_sample(prep_queue_len, duration)
        self.surgery_queue_stats.add_sample(surgery_queue_len, duration)
        self.recovery_queue_stats.add_sample(recovery_queue_len, duration)
    
    def update_resource_utilization(self, resource_name: str, busy_units: int, 
                                   capacity: int = 1, duration: float = 0.0):
        current_time = self.env.now
        
        if resource_name == 'prep':
            self.prep_utilization.total_time = current_time
            self.prep_utilization.capacity = capacity
            self.prep_utilization.busy_time += busy_units * duration
        elif resource_name == 'theatre':
            self.theatre_utilization.total_time = current_time
            self.theatre_utilization.capacity = capacity
            self.theatre_utilization.busy_time += busy_units * duration
        elif resource_name == 'recovery':
            self.recovery_utilization.total_time = current_time
            self.recovery_utilization.capacity = capacity
            self.recovery_utilization.busy_time += busy_units * duration
    
    def start_blocking(self, time: float):
        if self.current_blocking_start is None:
            self.current_blocking_start = time
    
    def end_blocking(self, time: float):
        if self.current_blocking_start is not None:
            self.blocking_events.append((self.current_blocking_start, time))
            self.current_blocking_start = None
    
    def finalize(self, end_time: float, prep_resource: simpy.Resource = None,
                surgery_resource: simpy.Resource = None,
                recovery_resource: simpy.Resource = None):
        if self.current_blocking_start is not None:
            self.blocking_events.append((self.current_blocking_start, end_time))
        
        duration = end_time - self.last_sample_time
        if duration > 0:
            if prep_resource is not None and surgery_resource is not None and recovery_resource is not None:
                prep_queue_len = len(prep_resource.queue)
                surgery_queue_len = len(surgery_resource.queue)
                recovery_queue_len = len(recovery_resource.queue)
                
                self.prep_queue_stats.add_sample(prep_queue_len, duration)
                self.surgery_queue_stats.add_sample(surgery_queue_len, duration)
                self.recovery_queue_stats.add_sample(recovery_queue_len, duration)
                
                prep_busy_units = prep_resource.count
                surgery_busy_units = surgery_resource.count
                recovery_busy_units = recovery_resource.count
                
                self.update_resource_utilization('prep', prep_busy_units, prep_resource.capacity, duration)
                self.update_resource_utilization('theatre', surgery_busy_units, surgery_resource.capacity, duration)
                self.update_resource_utilization('recovery', recovery_busy_units, recovery_resource.capacity, duration)
            else:
                self.prep_queue_stats.add_sample(0, duration)
                self.surgery_queue_stats.add_sample(0, duration)
                self.recovery_queue_stats.add_sample(0, duration)
        
        self.prep_utilization.total_time = end_time
        self.theatre_utilization.total_time = end_time
        self.recovery_utilization.total_time = end_time
    
    def get_blocking_statistics(self) -> Dict:
        if not self.blocking_events:
            return {
                'total_blocking_time': 0.0,
                'blocking_probability': 0.0,
                'num_blocking_events': 0,
                'avg_blocking_duration': 0.0
            }
        
        total_blocking = sum(end - start for start, end in self.blocking_events)
        total_time = self.theatre_utilization.total_time
        
        return {
            'total_blocking_time': total_blocking,
            'blocking_probability': (total_blocking / total_time * 100.0) if total_time > 0 else 0.0,
            'num_blocking_events': len(self.blocking_events),
            'avg_blocking_duration': total_blocking / len(self.blocking_events) if self.blocking_events else 0.0
        }
    
    def get_patient_statistics(self) -> Dict:
        completed_patients = [
            record for record in self.patient_records.values()
            if record.departure_time is not None
        ]
        
        if not completed_patients:
            return {
                'total_patients': len(self.patient_records),
                'completed_patients': 0,
                'avg_throughput_time': 0.0,
                'min_throughput_time': 0.0,
                'max_throughput_time': 0.0
            }
        
        throughput_times = [p.get_throughput_time() for p in completed_patients]
        
        return {
            'total_patients': len(self.patient_records),
            'completed_patients': len(completed_patients),
            'avg_throughput_time': sum(throughput_times) / len(throughput_times),
            'min_throughput_time': min(throughput_times),
            'max_throughput_time': max(throughput_times)
        }
    
    def generate_report(self) -> str:
        patient_stats = self.get_patient_statistics()
        blocking_stats = self.get_blocking_statistics()
        
        report = []
        report.append("=" * 70)
        report.append("HOSPITAL PATIENT FLOW SIMULATION - PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Patient statistics
        report.append("PATIENT STATISTICS:")
        report.append(f"  Total patients arrived: {patient_stats['total_patients']}")
        report.append(f"  Completed patients: {patient_stats['completed_patients']}")
        report.append(f"  Average throughput time: {patient_stats['avg_throughput_time']:.2f}")
        report.append(f"  Min throughput time: {patient_stats['min_throughput_time']:.2f}")
        report.append(f"  Max throughput time: {patient_stats['max_throughput_time']:.2f}")
        if self.urgent_throughputs:
            report.append(f"  Avg urgent throughput time: {sum(self.urgent_throughputs)/len(self.urgent_throughputs):.2f}")

        if self.routine_throughputs:
            report.append(f"  Avg routine throughput time: {sum(self.routine_throughputs)/len(self.routine_throughputs):.2f}")
        
        report.append("")
        
        # Queue statistics
        report.append("QUEUE STATISTICS:")
        report.append(f"  Average prep queue length: {self.prep_queue_stats.get_average_length():.2f}")
        report.append(f"  Average surgery queue length: {self.surgery_queue_stats.get_average_length():.2f}")
        report.append(f"  Average recovery queue length: {self.recovery_queue_stats.get_average_length():.2f}")
        report.append(f"  Max prep queue length: {self.prep_queue_stats.max_length}")
        report.append(f"  Max surgery queue length: {self.surgery_queue_stats.max_length}")
        report.append(f"  Max recovery queue length: {self.recovery_queue_stats.max_length}")
        report.append("")
        
        # Resource utilization
        report.append("RESOURCE UTILIZATION:")
        report.append(f"  Preparation rooms: {self.prep_utilization.get_utilization():.2f}%")
        report.append(f"  Operating theatre: {self.theatre_utilization.get_utilization():.2f}%")
        report.append(f"  Recovery rooms: {self.recovery_utilization.get_utilization():.2f}%")
        report.append("")
        
        # Blocking statistics
        report.append("OPERATING THEATRE BLOCKING:")
        report.append(f"  Total blocking time: {blocking_stats['total_blocking_time']:.2f}")
        report.append(f"  Blocking probability: {blocking_stats['blocking_probability']:.2f}%")
        report.append(f"  Number of blocking events: {blocking_stats['num_blocking_events']}")
        report.append(f"  Average blocking duration: {blocking_stats['avg_blocking_duration']:.2f}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


def monitoring_process(env: simpy.Environment, monitor: SystemMonitor,
                      prep_resource: simpy.Resource, surgery_resource: simpy.Resource,
                      recovery_resource: simpy.Resource, interval: float):
    while True:
        current_time = env.now
        duration = current_time - monitor.last_sample_time
        
        prep_queue_len = len(prep_resource.queue)
        surgery_queue_len = len(surgery_resource.queue)
        recovery_queue_len = len(recovery_resource.queue)
        
        monitor.sample_queue_lengths(prep_queue_len, surgery_queue_len, recovery_queue_len, duration)
        
        prep_busy_units = prep_resource.count
        surgery_busy_units = surgery_resource.count
        recovery_busy_units = recovery_resource.count
        
        monitor.update_resource_utilization('prep', prep_busy_units, prep_resource.capacity, duration)
        monitor.update_resource_utilization('theatre', surgery_busy_units, surgery_resource.capacity, duration)
        monitor.update_resource_utilization('recovery', recovery_busy_units, recovery_resource.capacity, duration)

	# Track probability that all recovery units are busy
        if recovery_busy_units == recovery_resource.capacity:
            monitor.time_all_recovery_busy += duration

        monitor.total_observed_time += duration

        monitor.last_sample_time = current_time
        yield env.timeout(interval)

