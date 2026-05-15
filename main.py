import sys
from typing import List
from models.person import Person
from engine.simulator import Simulator

def main():
    """CLI Interactiva para la Evolving Population simulation."""
    print("==================================================")
    print("🌍 BIENVENIDO A EVOLVING POPULATION SIMULATOR 🌍")
    print("==================================================")
    
    sim = Simulator(initial_females=500, initial_males=500)
    print("Población génesis creada: 1000 habitantes.\n")
    
    while True:
        print(f"\n--- AÑO ACTUAL: {int(sim.current_time_years)} | Población Viva: {len(sim.get_alive_population())} ---")
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
    print(f"Población inicial: 1000")
    print(f"Población final viva: {len(sim.get_alive_population())}")
    print(f"Total histórico nacimientos: {sim.total_births}")
    print(f"Total histórico fallecimientos: {sim.total_deaths}")

if __name__ == "__main__":
    main()
