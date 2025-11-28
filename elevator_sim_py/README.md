Elevator simulator (SimPy)

Files added:
- `simulator.py`: main script. Run `python simulator.py --csv sample_passengers.csv`.
- `passenger_loader.py`: CSV loader for passenger generation.
- `gui.py`: optional pygame GUI (simple stub).
- `sample_passengers.csv`: small example CSV.
- `requirements.txt`

CSV format:
- Header: `time,origin,destination,id` (id optional)
- `time` is the simulation time when the passenger appears at the origin floor.
- `origin` and `destination` are integer floors (1..N)

Quick start (Windows PowerShell):

Install dependencies (optional for GUI):
```
python -m pip install -r elevator_sim_py\requirements.txt
```

Run the simulator (no GUI required):
```
python elevator_sim_py\simulator.py --csv elevator_sim_py\sample_passengers.csv --floors 10 --until 60
```

Notes & assumptions:
- Single elevator, simple up/down sweep policy.
- Travel time between adjacent floors is 1 time unit; door time is 3 time units.
- Capacity defaults to 8.
- If `simpy` is not installed the script prints instructions to install it.

Next steps you might want:
- Extend `Elevator` decision policy (elevator scheduling algorithms).
- Improve GUI to show waiting queues and onboard passengers.
- Add multiple elevators.
