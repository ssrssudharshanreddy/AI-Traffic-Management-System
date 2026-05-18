# 🚦 SmartFlow — AI Traffic Management System

An intelligent, weight-based traffic signal controller that dynamically prioritizes lanes — with automatic emergency vehicle override — using a Python backend and an HTML frontend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Interactive CLI](#interactive-cli)
  - [Simulation Mode](#simulation-mode)
  - [CLI Commands](#cli-commands)
- [Vehicle Types & Weights](#vehicle-types--weights)
- [Algorithm](#algorithm)
- [Logging](#logging)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Overview

**SmartFlow** is an AI-driven traffic management system that simulates a four-way intersection (North, South, East, West). Instead of fixed timers, it uses a dynamic scoring algorithm to decide which lane gets a green signal each cycle — factoring in queue length, vehicle weight, and emergency status.

---

## Features

- 🚨 **Emergency Override** — Ambulances, Fire Trucks, and Police vehicles instantly jump to the front of priority
- ⚖️ **Weight-Based Scoring** — Heavier/larger vehicles (buses, trucks) contribute more to a lane's priority score
- 🔄 **Dynamic Cycle Control** — Each cycle clears at least 50% of the winning lane's queue
- 🖥️ **Interactive CLI** — Add vehicles, run cycles, and inspect intersection state in real time
- 🎲 **Automated Simulation** — Run randomized multi-cycle demos to observe the algorithm in action
- 📝 **Logging** — All events written to `traffic_log.txt` with timestamps
- 🎨 **Color-coded Terminal Output** — Green/red signals, emergency alerts, and stats rendered with ANSI colors

---

## Project Structure

```
AI-Traffic-Management-System/
├── python.py       # Core backend: controller, simulator, and CLI
└── index.html      # Frontend UI for the traffic system
```

---

## How It Works

Each traffic cycle follows these steps:

1. **Rank lanes** by a priority tuple: `(has_emergency, total_weight, queue_length)`
2. The **highest-scoring lane** receives a GREEN signal; all others turn RED
3. **50% of the winning lane's queue** is cleared (minimum 1 vehicle)
4. If an emergency vehicle is present, the cycle is flagged as an **Emergency Override**
5. Stats are updated and the state is logged

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- No third-party packages required — uses only the Python standard library

### Installation

```bash
git clone https://github.com/ssrssudharshanreddy/AI-Traffic-Management-System.git
cd AI-Traffic-Management-System
```

### Run the CLI

```bash
python python.py
```

### Run the Simulation directly

```bash
python python.py --simulate        # 10 cycles (default)
python python.py --simulate 20     # 20 cycles
```

---

## Usage

### Interactive CLI

Launch the interactive shell:

```bash
python python.py
```

You'll see the `SmartFlow>` prompt. Type `help` to list all commands.

### Simulation Mode

Run a fully automated randomized simulation:

```bash
# From CLI prompt:
SmartFlow> simulate
SmartFlow> simulate 15

# Or directly from terminal:
python python.py --simulate 10
```

### CLI Commands

| Command | Description |
|---|---|
| `add <direction> <vehicle_type>` | Add a vehicle to a lane |
| `cycle` | Run one traffic control cycle |
| `status` | Show current intersection state |
| `stats` | Show session statistics |
| `simulate [cycles]` | Run automated simulation (default: 10) |
| `clear <direction>` | Empty a specific lane |
| `help` | Show all available commands |
| `quit` / `exit` | Exit SmartFlow |

**Example session:**

```
SmartFlow> add North Ambulance
SmartFlow> add South Car
SmartFlow> add East Truck
SmartFlow> cycle
SmartFlow> status
```

---

## Vehicle Types & Weights

| Vehicle | Weight | Emergency |
|---|---|---|
| Car | 1 | No |
| Motorcycle | 1 | No |
| Truck | 3 | No |
| Bus | 5 | No |
| Ambulance | 100 | ✅ Yes |
| Fire Truck | 100 | ✅ Yes |
| Police | 100 | ✅ Yes |

Emergency vehicles are guaranteed the next green signal regardless of other lanes.

---

## Algorithm

Lane priority is determined by comparing a 3-tuple lexicographically:

```
score = (has_emergency, total_weight, queue_length)
```

- `has_emergency` (1 or 0) is evaluated first — any lane with an emergency vehicle wins immediately
- `total_weight` breaks ties between non-emergency lanes — heavier traffic gets priority
- `queue_length` is the final tiebreaker

On each cycle, **50% of the winning lane's vehicles** are cleared (rounded up, minimum 1), simulating a realistic green-phase flow rate.

---

## Logging

All activity is logged to `traffic_log.txt` in the working directory:

```
2025-01-01 12:00:00 | INFO | SmartFlow Controller initialized. All signals set to RED.
2025-01-01 12:00:01 | INFO | Vehicle ADDED → North: 🚨Ambulance(w=100)
2025-01-01 12:00:02 | INFO | Cycle 1 | GREEN → North | Reason: 🚨 EMERGENCY OVERRIDE | Cleared: 1 vehicle(s)
```

---

## Technologies Used

- **Python 3** — Core language
- **`dataclasses`** — Clean data modeling for `Vehicle` and `Lane`
- **`collections.deque`** — Efficient FIFO queue for each lane
- **`logging`** — Dual output (file + console)
- **ANSI escape codes** — Color-coded terminal UI
- **HTML/CSS/JS** — Frontend interface (`index.html`)

---

## Future Improvements

- [ ] REST API to connect the Python backend with the HTML frontend in real time
- [ ] WebSocket support for live intersection state updates in the browser
- [ ] Pedestrian crossing signals and countdown timers
- [ ] Multi-intersection network simulation
- [ ] Historical traffic analytics and visualization
- [ ] Machine learning model to predict and adapt to traffic patterns

---

## Author

**ssrssudharshanreddy** — [GitHub Profile](https://github.com/ssrssudharshanreddy)

---

*SmartFlow — because traffic shouldn't wait when it doesn't have to.*
