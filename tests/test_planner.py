# test_planner.py
import os
from env import GridEnvironment
from constraintext import ConstraintExtractor
from cfgen import CostFunctionGenerator, a_star_search, uniform_cost_function
from viz import visualize_grid, visualize_comparison, visualize_regions
from eval import evaluate_path, compare_paths

# Set your Groq API key
# os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # Replace with your actual API key

def test_small_grid():
    print("Testing with small grid...")
    
    # Create a simple 10x10 grid environment
    env = GridEnvironment(10, 10)
    
    # Add some obstacles
    for x in range(3, 6):
        env.add_obstacle(x, 4)
    env.add_obstacle(7, 2)
    env.add_obstacle(2, 7)
    
    # Define regions
    env.define_region_from_bounds('center', 3, 3, 6, 6)
    env.define_region('walls', [(0, y) for y in range(10)] + 
                             [(9, y) for y in range(10)] + 
                             [(x, 0) for x in range(10)] + 
                             [(x, 9) for x in range(10)])
    env.define_region_from_bounds('north', 0, 0, 9, 3)
    env.define_region_from_bounds('south', 0, 6, 9, 9)
    
    # Define start and goal
    start = (0, 0)
    goal = (9, 9)
    
    # Get instruction
    instruction = "Go to the goal while staying close to the walls and avoiding the center area."
    
    # Extract constraints - disable visualization by default
    extractor = ConstraintExtractor()
    constraints = extractor.extract_constraints(instruction, env, include_visualization=False)
    print(f"Extracted constraints: {constraints}")
    
    # Generate cost function
    generator = CostFunctionGenerator()
    cost_function = generator.generate_cost_function(constraints.dict(), env)
    
    # Find path with constraints
    constraint_path = a_star_search(env, start, goal, cost_function)
    print(f"Found constraint path with {len(constraint_path)} steps")
    
    # Find baseline path
    baseline_cost = uniform_cost_function(env)
    baseline_path = a_star_search(env, start, goal, baseline_cost)
    print(f"Found baseline path with {len(baseline_path)} steps")
    
    # Evaluate paths
    metrics = compare_paths(constraint_path, baseline_path, env, constraints.dict())
    print("Comparison metrics:")
    print(metrics)
    
    # Visualize results
    visualize_grid(env, constraint_path, start, goal, constraints.dict(),
                  title="Constraint-Based Path")
    
    visualize_comparison(env, constraint_path, baseline_path, start, goal, constraints.dict())
    
    return constraint_path, baseline_path, metrics

def test_auto_regions():
    print("Testing with automatic region identification...")
    
    # Create a more complex environment
    env = create_complex_environment(20, 20)
    
    # Define start and goal
    start = (2, 2)
    goal = (17, 17)
    
    # Automatically identify regions - use text-only method to avoid token limits
    extractor = ConstraintExtractor()
    try:
        # First try with image-based method
        regions = extractor.identify_regions(env)
    except Exception as e:
        print(f"Image-based region identification failed: {e}")
        print("Falling back to text-only region identification...")
        # Fall back to text-only method
        regions = extractor.identify_regions_text_only(env)
    
    # Add identified regions to environment
    for region_name, coords in regions.items():
        env.regions[region_name] = coords
    
    # Print identified regions
    print("\nAutomatically identified regions:")
    for region_name, coords in regions.items():
        print(f"- {region_name}: {len(coords)} cells")
        if hasattr(env, 'region_descriptions') and region_name in env.region_descriptions:
            print(f"  Description: {env.region_descriptions[region_name]}")
    
    # Visualize the identified regions
    visualize_regions(env, start, goal, title="Automatically Identified Regions")
    
    # Get instruction
    instruction = "Navigate to the goal efficiently while avoiding narrow passages and staying away from obstacles."
    
    # Extract constraints
    constraints = extractor.extract_constraints(instruction, env, include_visualization=False)
    print(f"\nExtracted constraints: {constraints}")
    
    # Generate cost function
    generator = CostFunctionGenerator()
    cost_function = generator.generate_cost_function(constraints.dict(), env)
    
    # Find path with constraints
    constraint_path = a_star_search(env, start, goal, cost_function)
    print(f"\nFound constraint path with {len(constraint_path)} steps")
    
    # Find baseline path
    baseline_cost = uniform_cost_function(env)
    baseline_path = a_star_search(env, start, goal, baseline_cost)
    print(f"Found baseline path with {len(baseline_path)} steps")
    
    # Evaluate paths
    metrics = compare_paths(constraint_path, baseline_path, env, constraints.dict())
    print("\nComparison metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Visualize results
    visualize_grid(env, constraint_path, start, goal, constraints.dict(),
                  title="Constraint-Based Path with Auto-Identified Regions")
    
    visualize_comparison(env, constraint_path, baseline_path, start, goal, constraints.dict())
    
    return constraint_path, baseline_path, metrics

def create_complex_environment(width, height):
    """Create a more complex environment with obstacles"""
    env = GridEnvironment(width, height)
    
    # Add some walls/obstacles
    
    # Vertical wall with a gap
    for y in range(5, 15):
        if y != 10:  # Gap at y=10
            env.add_obstacle(10, y)
    
    # Horizontal wall with a gap
    for x in range(5, 15):
        if x != 7:  # Gap at x=7
            env.add_obstacle(x, 12)
    
    # Some random obstacles
    obstacles = [
        (3, 3), (4, 7), (7, 15), (15, 5), (17, 8),
        (5, 17), (14, 14), (8, 8), (12, 3), (3, 12)
    ]
    
    for x, y in obstacles:
        env.add_obstacle(x, y)
    
    return env

if __name__ == "__main__":
    # Run the original test
    # test_small_grid()
    
    # Run the automatic region identification test
    test_auto_regions()