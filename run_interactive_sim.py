import argparse

from engine.simulator import Simulator


def run_once(months: int, sim: Simulator) -> None:
    sim.current_year_logs.clear()
    births_before = sim.total_births
    deaths_before = sim.total_deaths
    sim.run(months / 12.0)
    for e in sim.current_year_logs:
        print(" -", e)
    alive = len(sim.population_alive)
    print(
        "poblacion=", alive,
        "births=", sim.total_births - births_before,
        "deaths=", sim.total_deaths - deaths_before,
    )


def interactive_loop(sim: Simulator) -> None:
    while True:
        try:
            unit = input("Unidad (m=meses, a=anios, 0=salir): ").strip().lower()
            if unit == "0":
                break
            if unit not in {"m", "a"}:
                print("Unidad invalida")
                continue
            raw = input("Cantidad a avanzar: ").strip()
            amount = int(raw)
            if amount <= 0:
                print("Cantidad invalida")
                continue
            months = amount if unit == "m" else amount * 12
        except Exception:
            print("Valor invalido")
            continue
        run_once(months, sim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=0)
    parser.add_argument("--female", type=int, default=100)
    parser.add_argument("--male", type=int, default=100)
    args = parser.parse_args()

    sim = Simulator(initial_females=args.female, initial_males=args.male)
    print("inicio: poblacion=", len(sim.population_alive))

    if args.months and args.months > 0:
        run_once(args.months, sim)
    else:
        interactive_loop(sim)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
