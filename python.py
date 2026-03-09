"""
SmartFlow - Intelligent Traffic Controller
Dynamic, weight-based traffic management prioritizing emergency response.
"""

import random
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    filename="traffic_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(console)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
VEHICLE_WEIGHTS: Dict[str, int] = {
    "Car":         1,
    "Motorcycle":  1,
    "Truck":       3,
    "Bus":         5,
    "Ambulance":   100,
    "Fire Truck":  100,
    "Police":      100,
}

EMERGENCY_TYPES = {"Ambulance", "Fire Truck", "Police"}

DIRECTIONS = ["North", "South", "East", "West"]

COLORS = {
    "RED":    "\033[91m",
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE":   "\033[94m",
    "CYAN":   "\033[96m",
    "WHITE":  "\033[97m",
    "BOLD":   "\033[1m",
    "DIM":    "\033[2m",
    "RESET":  "\033[0m",
}

def c(color: str, text: str) -> str:
    """Colorize terminal text."""
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class Vehicle:
    vehicle_type: str
    weight: int = field(init=False)
    is_emergency: bool = field(init=False)
    arrival_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.vehicle_type not in VEHICLE_WEIGHTS:
            raise ValueError(f"Unknown vehicle type: '{self.vehicle_type}'. "
                             f"Valid types: {list(VEHICLE_WEIGHTS.keys())}")
        self.weight = VEHICLE_WEIGHTS[self.vehicle_type]
        self.is_emergency = self.vehicle_type in EMERGENCY_TYPES

    def __repr__(self):
        icon = "🚨" if self.is_emergency else "🚗"
        return f"{icon}{self.vehicle_type}(w={self.weight})"


@dataclass
class Lane:
    direction: str
    queue: deque = field(default_factory=deque)

    def add_vehicle(self, vehicle: Vehicle):
        self.queue.append(vehicle)

    def remove_vehicles(self, count: int) -> List[Vehicle]:
        """Remove `count` vehicles from the front and return them."""
        removed = []
        for _ in range(min(count, len(self.queue))):
            removed.append(self.queue.popleft())
        return removed

    @property
    def has_emergency(self) -> bool:
        return any(v.is_emergency for v in self.queue)

    @property
    def total_weight(self) -> int:
        return sum(v.weight for v in self.queue)

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    def score(self) -> Tuple[int, int, int]:
        """Returns (has_emergency, total_weight, queue_length) for ranking."""
        return (int(self.has_emergency), self.total_weight, self.queue_length)

    def __repr__(self):
        return f"Lane({self.direction}, vehicles={self.queue_length}, weight={self.total_weight})"


# ─── Core Controller ──────────────────────────────────────────────────────────
class SmartFlowController:
    def __init__(self):
        self.lanes: Dict[str, Lane] = {d: Lane(d) for d in DIRECTIONS}
        self.cycle_count = 0
        self.total_vehicles_cleared = 0
        self.emergency_overrides = 0
        self.signal_states: Dict[str, str] = {d: "RED" for d in DIRECTIONS}
        self.active_green: Optional[str] = None
        log.info("SmartFlow Controller initialized. All signals set to RED.")

    # ── Input Methods ─────────────────────────────────────────────────────────
    def add_vehicle(self, direction: str, vehicle_type: str) -> Vehicle:
        """Add a vehicle to a lane queue."""
        direction = direction.capitalize()
        if direction not in self.lanes:
            raise ValueError(f"Invalid direction '{direction}'. Choose from: {DIRECTIONS}")
        vehicle = Vehicle(vehicle_type)
        self.lanes[direction].add_vehicle(vehicle)
        log.info(f"Vehicle ADDED → {direction} lane: {vehicle}")
        return vehicle

    def bulk_add(self, entries: List[Tuple[str, str]]):
        """Add multiple vehicles. entries = [(direction, vehicle_type), ...]"""
        for direction, vtype in entries:
            self.add_vehicle(direction, vtype)

    # ── Routing Algorithm ─────────────────────────────────────────────────────
    def _rank_lanes(self) -> List[Tuple[str, Lane]]:
        """
        Rank all lanes by priority tuple: (has_emergency, total_weight, queue_length).
        Higher tuple = higher priority (compared lexicographically).
        """
        active = [(d, lane) for d, lane in self.lanes.items() if lane.queue_length > 0]
        return sorted(active, key=lambda x: x[1].score(), reverse=True)

    def run_cycle(self) -> Optional[str]:
        """
        Execute one traffic control cycle.
        Returns the direction given Green Phase, or None if all lanes empty.
        """
        self.cycle_count += 1
        ranked = self._rank_lanes()

        if not ranked:
            log.info(f"Cycle {self.cycle_count}: All lanes EMPTY. No signal change.")
            self._set_all_red()
            return None

        winner_dir, winner_lane = ranked[0]
        is_override = winner_lane.has_emergency

        # Set signals
        self._set_green(winner_dir)

        # Determine vehicles to clear: at least 50% of queue, min 1
        clear_count = max(1, (winner_lane.queue_length + 1) // 2)
        cleared = winner_lane.remove_vehicles(clear_count)
        self.total_vehicles_cleared += len(cleared)

        if is_override:
            self.emergency_overrides += 1
            reason = "🚨 EMERGENCY OVERRIDE"
        else:
            reason = "📊 Highest weighted score"

        log.info(
            f"Cycle {self.cycle_count} | GREEN → {winner_dir} | "
            f"Reason: {reason} | "
            f"Score: {winner_lane.score() if winner_lane.queue_length > 0 else 'N/A (pre-clear)'} | "
            f"Cleared: {len(cleared)} vehicle(s) | "
            f"Remaining: {winner_lane.queue_length}"
        )

        return winner_dir

    def _set_green(self, direction: str):
        for d in DIRECTIONS:
            self.signal_states[d] = "GREEN" if d == direction else "RED"
        self.active_green = direction

    def _set_all_red(self):
        for d in DIRECTIONS:
            self.signal_states[d] = "RED"
        self.active_green = None

    # ── Display Methods ───────────────────────────────────────────────────────
    def display_intersection(self):
        """Print a visual representation of the current intersection state."""
        print()
        print(c("BOLD", "═" * 60))
        print(c("CYAN", c("BOLD", "       🚦 SMARTFLOW INTERSECTION STATUS")))
        print(c("BOLD", "═" * 60))

        for direction in DIRECTIONS:
            lane = self.lanes[direction]
            state = self.signal_states[direction]
            color = "GREEN" if state == "GREEN" else "RED"
            signal_dot = c(color, f"[{state}]")
            emergency_tag = c("RED", " ⚠ EMERGENCY") if lane.has_emergency else ""
            vehicles_str = ", ".join(str(v) for v in lane.queue) if lane.queue else c("DIM", "empty")
            print(f"  {c('BOLD', direction.ljust(6))} {signal_dot.ljust(20)}"
                  f" | W:{lane.total_weight:>4} | Q:{lane.queue_length:>2} | {vehicles_str}{emergency_tag}")

        print(c("BOLD", "─" * 60))
        print(f"  Cycle: {c('YELLOW', str(self.cycle_count))} | "
              f"Cleared: {c('GREEN', str(self.total_vehicles_cleared))} | "
              f"Emergency Overrides: {c('RED', str(self.emergency_overrides))}")
        print(c("BOLD", "═" * 60))
        print()

    def display_stats(self):
        """Print final run statistics."""
        print()
        print(c("BOLD", "═" * 60))
        print(c("CYAN", c("BOLD", "       📊 SMARTFLOW SESSION STATISTICS")))
        print(c("BOLD", "═" * 60))
        print(f"  Total Cycles Run       : {c('YELLOW', str(self.cycle_count))}")
        print(f"  Total Vehicles Cleared : {c('GREEN', str(self.total_vehicles_cleared))}")
        print(f"  Emergency Overrides    : {c('RED', str(self.emergency_overrides))}")
        remaining = sum(l.queue_length for l in self.lanes.values())
        print(f"  Vehicles Still Queued  : {c('WHITE', str(remaining))}")
        print(c("BOLD", "═" * 60))
        print()


# ─── Simulation Mode ──────────────────────────────────────────────────────────
class SmartFlowSimulator:
    """Randomized multi-cycle simulation to demonstrate algorithm viability."""

    SPAWN_WEIGHTS = [
        ("Car", 50), ("Motorcycle", 15), ("Truck", 10),
        ("Bus", 10), ("Ambulance", 5), ("Fire Truck", 5), ("Police", 5),
    ]

    def __init__(self, cycles: int = 10, max_arrivals_per_cycle: int = 4, delay: float = 0.6):
        self.cycles = cycles
        self.max_arrivals = max_arrivals_per_cycle
        self.delay = delay
        self.controller = SmartFlowController()
        self._vehicle_pool = []
        for vtype, weight in self.SPAWN_WEIGHTS:
            self._vehicle_pool.extend([vtype] * weight)

    def _random_vehicle(self) -> str:
        return random.choice(self._vehicle_pool)

    def run(self):
        print()
        print(c("BOLD", "╔" + "═" * 58 + "╗"))
        print(c("BOLD", "║") + c("CYAN", c("BOLD",
              "   🚦  SMARTFLOW — AUTOMATED SIMULATION MODE  🚦".center(58))) + c("BOLD", "║"))
        print(c("BOLD", "╚" + "═" * 58 + "╝"))
        print(f"  Simulating {c('YELLOW', str(self.cycles))} cycles with up to "
              f"{c('YELLOW', str(self.max_arrivals))} vehicle arrivals each.\n")
        time.sleep(0.5)

        for cycle in range(1, self.cycles + 1):
            print(c("BOLD", f"\n── PRE-CYCLE {cycle} ARRIVALS ──────────────────────────────"))
            arrivals = random.randint(0, self.max_arrivals)
            for _ in range(arrivals):
                direction = random.choice(DIRECTIONS)
                vtype = self._random_vehicle()
                vehicle = self.controller.add_vehicle(direction, vtype)
                emergency_note = c("RED", " ← EMERGENCY!") if vehicle.is_emergency else ""
                print(f"  Arriving → {direction}: {vehicle}{emergency_note}")

            self.controller.display_intersection()
            print(c("BOLD", f"── RUNNING CYCLE {cycle} ──────────────────────────────────"))
            self.controller.run_cycle()
            time.sleep(self.delay)

        # Final state
        print(c("BOLD", "\n── FINAL INTERSECTION STATE ─────────────────────────"))
        self.controller.display_intersection()
        self.controller.display_stats()
        log.info("Simulation complete.")


# ─── Interactive CLI ──────────────────────────────────────────────────────────
class SmartFlowCLI:
    def __init__(self):
        self.controller = SmartFlowController()

    def print_help(self):
        print(c("CYAN", """
  Commands:
    add <direction> <vehicle_type>  — Add a vehicle to a lane
    cycle                           — Run one traffic control cycle
    status                          — Show current intersection state
    stats                           — Show session statistics
    simulate [cycles]               — Run automated simulation (default 10 cycles)
    clear <direction>               — Empty a lane
    help                            — Show this help
    quit / exit                     — Exit SmartFlow

  Directions : North, South, East, West
  Vehicle Types: Car, Motorcycle, Truck, Bus, Ambulance, Fire Truck, Police
"""))

    def run(self):
        print()
        print(c("BOLD", "╔" + "═" * 58 + "╗"))
        print(c("BOLD", "║") + c("CYAN", c("BOLD",
              "   🚦  SMARTFLOW INTELLIGENT TRAFFIC CONTROLLER   ".center(58))) + c("BOLD", "║"))
        print(c("BOLD", "╚" + "═" * 58 + "╝"))
        print(c("DIM", "  Type 'help' for commands or 'simulate' for auto-demo.\n"))

        while True:
            try:
                raw = input(c("CYAN", "SmartFlow> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("YELLOW", "\n  Shutting down SmartFlow. Goodbye!"))
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit"):
                print(c("YELLOW", "  Shutting down SmartFlow. Goodbye!"))
                break

            elif cmd == "help":
                self.print_help()

            elif cmd == "status":
                self.controller.display_intersection()

            elif cmd == "stats":
                self.controller.display_stats()

            elif cmd == "cycle":
                result = self.controller.run_cycle()
                if result:
                    state = self.controller.signal_states[result]
                    print(c("GREEN", f"  ✔ Green Phase → {result}"))
                else:
                    print(c("YELLOW", "  ⚠ All lanes empty. No cycle executed."))
                self.controller.display_intersection()

            elif cmd == "add":
                if len(parts) < 3:
                    print(c("RED", "  Usage: add <direction> <vehicle_type>"))
                    print(c("DIM", "  Example: add North Ambulance"))
                    continue
                direction = parts[1]
                vehicle_type = " ".join(parts[2:])  # Handles "Fire Truck"
                try:
                    vehicle = self.controller.add_vehicle(direction, vehicle_type)
                    tag = c("RED", " 🚨 EMERGENCY DETECTED!") if vehicle.is_emergency else ""
                    print(c("GREEN", f"  ✔ Added {vehicle} to {direction} lane.") + tag)
                except ValueError as e:
                    print(c("RED", f"  ✘ Error: {e}"))

            elif cmd == "clear":
                if len(parts) < 2:
                    print(c("RED", "  Usage: clear <direction>"))
                    continue
                direction = parts[1].capitalize()
                if direction not in self.controller.lanes:
                    print(c("RED", f"  ✘ Unknown direction: {direction}"))
                else:
                    count = self.controller.lanes[direction].queue_length
                    self.controller.lanes[direction].queue.clear()
                    print(c("YELLOW", f"  Cleared {count} vehicle(s) from {direction} lane."))

            elif cmd == "simulate":
                cycles = 10
                if len(parts) > 1:
                    try:
                        cycles = int(parts[1])
                        if cycles < 1:
                            raise ValueError
                    except ValueError:
                        print(c("RED", "  ✘ Invalid cycle count. Using default (10)."))
                        cycles = 10
                sim = SmartFlowSimulator(cycles=cycles)
                sim.run()

            else:
                print(c("RED", f"  ✘ Unknown command: '{cmd}'. Type 'help' for usage."))


# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        sim = SmartFlowSimulator(cycles=cycles)
        sim.run()
    else:
        cli = SmartFlowCLI()
        cli.run()


if __name__ == "__main__":
    main()
