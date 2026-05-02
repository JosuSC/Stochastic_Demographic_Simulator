from typing import List
from models.person import Person
from engine.simulator import Simulator

def main():
    """Entry point for the Evolving Population simulation."""
    print("Iniciando simulación demográfica de 100 años...")
    
    # 100 años = 1200 meses
    total_months = 1200
    sim = Simulator(initial_females=500, initial_males=500)
    
    sim.run(total_months)
    
    print("\nSimulación completada.")
    print(f"Población inicial: 1000")
    print(f"Población final viva: {len(sim.get_alive_population())}")
    print(f"Total nacimientos: {sim.total_births}")
    print(f"Total fallecimientos: {sim.total_deaths}")

if __name__ == "__main__":
    main()
