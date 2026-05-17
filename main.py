import sys
from typing import List
from pathlib import Path
from models.person import Person
from engine.simulator import Simulator


def interactive_main(initial_females: int = 500, initial_males: int = 500) -> None:
    """CLI Interactiva para la Evolving Population simulation."""
    print("==================================================")
    print("🌍 BIENVENIDO A EVOLVING POPULATION SIMULATOR 🌍")
    print("==================================================")
    
    sim = Simulator(initial_females=initial_females, initial_males=initial_males)
    print(f"Población génesis creada: {len(sim.population_alive)} habitantes.\n")
    
    while True:
        print(f"\n--- AÑO ACTUAL: {int(sim.current_time_years)} | Población Viva: {len(sim.population_alive)} ---")
        print("Comandos disponibles:")
        print("  [1] o 'n'   -> Avanzar 1 año")
        print("  [x]         -> Avanzar 'X' cantidad de años (ejemplo: '10' avanza una década)")
        print("  [q] o 'ext' -> Terminar simulación (Quit)")
        
        user_input = input(">> ¿Tu orden, arquitecto?: ").strip().lower()
        
        if user_input in ('q', 'ext', 'exit', 'quit'):
            break
        
        años_a_simular = 1
        if user_input.isdigit():
            años_a_simular = int(user_input)
        elif user_input == 'n':
            años_a_simular = 1
            
        for y in range(años_a_simular):
            sim.current_year_logs.clear() # Limpiar logs del año viejo
            sim.run(1.0) # Correr 1 año
            
            # Imprimir el reporte del año entero si es paso a paso o si saltó años
            year_number = int(sim.current_time_years)
            print(f"\n\n================ REPORTE DEL AÑO {year_number} ================")
            if len(sim.current_year_logs) == 0:
                print("  Tranquilidad pacífica. Nada de mayor relevancia fuera de la monotonía.")
            else:
                for log in sim.current_year_logs:
                    print("  -", log)
                    
            print(f"--------------------------------------------------")
            print(f"RESUMEN ANUAL -> Nacimientos Totales Históricos: {sim.total_births} | Fallecimientos Totales Históricos: {sim.total_deaths}")
            
    print("\nSimulación abortada por el Arquitecto.")
    print("========== ESTADO FINAL DEL MUNDO ==========")
    print(f"Año alcanzado: {int(sim.current_time_years)}")
    print(f"Población inicial: {len(sim.population_alive) + sim.total_deaths}")
    print(f"Población final viva: {len(sim.population_alive)}")
    print(f"Total histórico nacimientos: {sim.total_births}")
    print(f"Total histórico fallecimientos: {sim.total_deaths}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: puede lanzar la CLI interactiva o ejecutar análisis estadístico."""
    import argparse

    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--analysis", action="store_true", help="Run statistical analysis flow and export CSVs")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs for analysis (when --analysis used)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for analysis")
    parser.add_argument("--raw-output", type=str, default="simulation_runs.csv", help="CSV path for raw runs")
    parser.add_argument("--stats-output", type=str, default="simulation_statistics.csv", help="CSV path for statistics")
    parser.add_argument("--female", type=int, default=500, help="Initial female population for interactive mode")
    parser.add_argument("--male", type=int, default=500, help="Initial male population for interactive mode")

    args = parser.parse_args(argv)

    if args.analysis:
        # Ejecutar el flujo de análisis desde run_analysis
        from run_analysis import main as run_analysis_main

        return run_analysis_main(num_runs=args.runs, seed=args.seed, raw_output=Path(args.raw_output), stats_output=Path(args.stats_output))

    # Modo interactivo
    interactive_main(initial_females=args.female, initial_males=args.male)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
