import datetime
from full_day_simulation import FullDayTradingSimulation

def main():
    """Main function to run a single cycle of the simulation"""
    print("🚀 Starting UFO Forex Agent v3 - SINGLE CYCLE TEST")

    try:
        # Create a simulation instance
        simulation = FullDayTradingSimulation(datetime.datetime(2025, 8, 4))

        # Run only the daily planning phase and one cycle
        simulation.log_event("\n" + "="*60)
        simulation.log_event("📈 PHASE 0: Daily Planning")
        simulation.log_event("="*60)
        initial_price_data = simulation.collect_market_data()
        initial_ufo_data = simulation.calculate_ufo_indicators(initial_price_data)
        initial_economic_events = simulation.get_economic_events()
        simulation.synthetic_portfolio_definition = simulation.trader.formulate_daily_hypothesis(initial_ufo_data, initial_economic_events)
        simulation.log_event(f"📊 Daily Hypothesis: {simulation.synthetic_portfolio_definition}")

        # Run a single cycle
        current_time = datetime.datetime(simulation.simulation_date.year, simulation.simulation_date.month, simulation.simulation_date.day, 8, 0)
        simulation.simulate_single_cycle(current_time)

        print(f"\n✅ Single cycle test completed successfully!")

    except Exception as e:
        print(f"\n💥 Single cycle test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
