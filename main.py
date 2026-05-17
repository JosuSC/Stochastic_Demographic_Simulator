"""
Small CLI for the population simulation.

Usage:
    python main.py                     # Interactive mode
    python main.py --months 120        # Run 120 months non-interactively
    python main.py --female 500 --male 500
"""

import sys
from engine.simulator import Simulator


def main():
    """Run the population simulator from the command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Evolving Population Simulator")
    parser.add_argument("--months", type=int, default=0, help="Months to simulate (0 = interactive)")
    parser.add_argument("--female", type=int, default=500, help="Initial female population")
    parser.add_argument("--male", type=int, default=500, help="Initial male population")
    args = parser.parse_args()

    print("=" * 60)
    print("  EVOLVING POPULATION SIMULATOR")
    print("=" * 60)

    sim = Simulator(initial_females=args.female, initial_males=args.male)
    print(f"Genesis population created: {args.female}F + {args.male}M = {len(sim.alive_ids)} people\n")

    if args.months and args.months > 0:
        # Run straight through and print a short summary at the end.
        sim.run(args.months / 12.0)
        stats = sim.get_statistics()
        print(f"Simulated {args.months} months ({args.months/12:.1f} years)")
        print(f"Population: {stats['population']} | Births: {stats['births']} | Deaths: {stats['deaths']}")
        return 0

    # Interactive loop.
    while True:
        stats = sim.get_statistics()
        print(f"\n--- Year {int(stats['time_years'])} | Population: {stats['population']} ---")
        print("Commands: [1/n] advance 1/N years | [q] quit")

        user_input = input(">> ").strip().lower()
        if user_input in ('q', 'exit', 'quit'):
            break

        years = 1
        if user_input.isdigit():
            years = int(user_input)

        for y in range(years):
            sim.current_year_logs.clear()
            sim.run(1.0)

        # Print the last few events from the most recent year.
        if sim.current_year_logs:
            for log in sim.current_year_logs[-10:]:  # Keep the output readable.
                print("  -", log)
            if len(sim.current_year_logs) > 10:
                print(f"  ... and {len(sim.current_year_logs) - 10} more events")

        stats = sim.get_statistics()
        print(f"  Population: {stats['population']} | M: {stats['males']} | F: {stats['females']}")
        print(f"  Births: {sim.total_births} | Deaths: {sim.total_deaths} | Couples: {stats['couples']}")

    print("\nSimulation ended by user.")
    stats = sim.get_statistics()
    print(f"Final year: {int(stats['time_years'])}")
    print(f"Final population: {stats['population']}")
    print(f"Total births: {sim.total_births}")
    print(f"Total deaths: {sim.total_deaths}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
