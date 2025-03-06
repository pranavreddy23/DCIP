#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiments import ExperimentRunner
from core.region_segmentation import RegionSegmenter
from core.cfgen import a_star_search, uniform_cost_function, CostFunctionGenerator
from utils.map_generator import ChallengeMapGenerator, generate_random_positions
from utils.viz import visualize_node_expansion, visualize_paths, visualize_regions, visualize_grid, visualize_cost_landscape

def run_efficiency_experiments():
    """Run experiments focused on search efficiency"""
    # Create timestamp for log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(log_dir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "regions"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "paths"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "visualizations"), exist_ok=True)
    
    # Create a single environment and run the experiment
    print("\nCreating environment and running experiment...")
    
    # Generate environment - increase size for more challenging scenarios
    width, height = 60, 60  # Increased from 40x40 to 60x60
    print(f"Generating cluttered environment of size {width}x{height}...")
    env = ChallengeMapGenerator.generate_cluttered(width, height)
    
    # Visualize the raw map first
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    plt.title(f"Cluttered Environment ({width}x{height})")
    plt.savefig(os.path.join(log_dir, "maps", "environment.png"))
    plt.show()  # Show the map immediately
    
    # Generate random start/goal positions with greater distance
    print("Generating random start/goal positions...")
    start, goal = generate_random_positions(env, min_distance=max(width, height) // 2)  # Increased distance
    print(f"Start: {start}, Goal: {goal}")
    
    # Visualize map with start and goal
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    plt.title(f"Environment with Start and Goal")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "maps", "environment_with_positions.png"))
    plt.show()  # Show the map with start/goal
    
    # Segment regions
    print("Segmenting regions...")
    segmenter = RegionSegmenter(method="watershed")
    env.regions = segmenter.segment_regions(env)
    
    # Visualize regions
    print("Visualizing regions...")
    visualize_regions(env, save_path=os.path.join(log_dir, "regions", "regions.png"))
    # Don't show regions immediately, just save them
    
    # Create uniform cost function for baseline
    uniform_cost = uniform_cost_function(env)
    
    # Run baseline planner
    print("Running baseline planner...")
    baseline_path, baseline_cost, baseline_expanded = a_star_search(
        env, start, goal, uniform_cost, track_expanded=True)
    
    # Visualize baseline path
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    if baseline_path:
        path_x, path_y = zip(*baseline_path)
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Baseline Path')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    plt.title(f"Baseline Path (Nodes Expanded: {len(baseline_expanded)})")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "paths", "baseline_path.png"))
    plt.show()  # Show baseline path
    
    # Choose an instruction focused on efficiency
    instruction = "Navigate to the goal efficiently while avoiding obstacles and taking the most direct route."
    print(f"Using instruction: {instruction}")
    
    # Generate constraints - Fix the method name
    print("Generating constraints...")
    cfgen = CostFunctionGenerator()
    # The correct method from your original code
    constraints = cfgen.extractor.extract_constraints(instruction, env)
    print(f"Extracted constraints: {constraints}")
    
    # Visualize grid with constraints
    print("Visualizing grid with constraints...")
    visualize_grid(env, baseline_path, start, goal, constraints, 
                  title="Environment with Constraints",
                  save_path=os.path.join(log_dir, "visualizations", "grid_with_constraints.png"))
    # Don't show constraints immediately, just save them
    
    # Generate constrained cost function with efficiency focus
    print("Generating constrained cost function...")
    # Use the correct parameter order: environment, instruction
    constrained_cost = cfgen.generate_cost_function(env, instruction, start, goal)
    
    # Enhance the cost function to improve search efficiency
    def enhanced_cost_function(x1, y1, x2, y2):
        # Get base cost from the constrained cost function
        base_cost = constrained_cost(x1, y1, x2, y2)
        
        # Add a heuristic bias toward the goal to improve search efficiency
        goal_direction = (goal[0] - x2, goal[1] - y2)
        current_direction = (x2 - x1, y2 - y1)
        
        # Calculate dot product to see if we're moving toward the goal
        dot_product = goal_direction[0] * current_direction[0] + goal_direction[1] * current_direction[1]
        
        # If moving toward goal, reduce cost slightly
        if dot_product > 0:
            efficiency_factor = 0.9  # 10% reduction for moving toward goal
        else:
            efficiency_factor = 1.1  # 10% increase for moving away from goal
            
        return base_cost * efficiency_factor
    
    # Visualize cost landscape
    print("Visualizing cost landscape...")
    visualize_cost_landscape(env, enhanced_cost_function, start, goal, instruction,
                           save_path=os.path.join(log_dir, "visualizations", "cost_landscape.png"))
    # Don't show cost landscape immediately, just save it
    
    # Run constrained planner with improved efficiency
    print("Running constrained planner with focus on search efficiency...")
    constrained_path, constrained_cost_value, constrained_expanded = a_star_search(
        env, start, goal, enhanced_cost_function, track_expanded=True)
    
    # Calculate path smoothness
    def calculate_path_smoothness(path):
        if not path or len(path) < 3:
            return 1.0  # Perfect smoothness for short paths
        
        angles = []
        for i in range(1, len(path) - 1):
            # Get vectors
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            
            # Calculate dot product
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            
            # Calculate magnitudes
            mag1 = (v1[0]**2 + v1[1]**2)**0.5
            mag2 = (v2[0]**2 + v2[1]**2)**0.5
            
            # Calculate angle (in radians)
            if mag1 * mag2 == 0:
                angle = 0
            else:
                angle = np.arccos(min(1, max(-1, dot_product / (mag1 * mag2))))
            
            angles.append(angle)
        
        # Average angle change (lower is smoother)
        avg_angle = sum(angles) / len(angles) if angles else 0
        
        # Normalize to [0, 1] where 1 is perfectly smooth
        smoothness = 1 - (avg_angle / np.pi)
        return smoothness
    
    baseline_smoothness = calculate_path_smoothness(baseline_path)
    constrained_smoothness = calculate_path_smoothness(constrained_path)
    
    # Calculate constraint compliance
    def calculate_constraint_compliance(path, constraints):
        if not path or not constraints:
            return {
                "proximity": 0.0,
                "avoidance": 0.0,
                "preference": 0.0,
                "overall": 0.0
            }
        
        # Initialize compliance metrics
        proximity_compliance = 0.0
        avoidance_compliance = 0.0
        preference_compliance = 0.0
        
        # Count compliant points
        total_points = len(path)
        proximity_points = 0
        avoidance_points = 0
        preference_points = 0
        
        # Check each point in the path
        for x, y in path:
            # Check proximity constraints
            if 'proximity' in constraints and constraints['proximity']:
                is_compliant = True
                for region, distance in constraints['proximity'].items():
                    if region in env.regions:
                        region_points = env.regions[region]
                        if region_points:
                            min_dist = min(abs(x - rx) + abs(y - ry) for rx, ry in region_points)
                            if min_dist > distance:
                                is_compliant = False
                                break
                if is_compliant:
                    proximity_points += 1
            else:
                proximity_points = total_points  # No constraints means full compliance
            
            # Check avoidance constraints
            if 'avoidance' in constraints and constraints['avoidance']:
                is_compliant = True
                for region, _ in constraints['avoidance'].items():
                    if region in env.regions:
                        if (x, y) in env.regions[region]:
                            is_compliant = False
                            break
                if is_compliant:
                    avoidance_points += 1
            else:
                avoidance_points = total_points
            
            # Check preference constraints
            if 'preference' in constraints and constraints['preference']:
                is_compliant = False
                for region, _ in constraints['preference'].items():
                    if region in env.regions:
                        if (x, y) in env.regions[region]:
                            is_compliant = True
                            break
                if is_compliant:
                    preference_points += 1
            else:
                preference_points = total_points
        
        # Calculate compliance ratios
        proximity_compliance = proximity_points / total_points if total_points > 0 else 0.0
        avoidance_compliance = avoidance_points / total_points if total_points > 0 else 0.0
        preference_compliance = preference_points / total_points if total_points > 0 else 0.0
        
        # Calculate overall compliance (average of all types)
        overall_compliance = (proximity_compliance + avoidance_compliance + preference_compliance) / 3
        
        return {
            "proximity": proximity_compliance,
            "avoidance": avoidance_compliance,
            "preference": preference_compliance,
            "overall": overall_compliance
        }
    
    baseline_compliance = calculate_constraint_compliance(baseline_path, constraints)
    constrained_compliance = calculate_constraint_compliance(constrained_path, constraints)
    
    # Visualize paths side by side
    print("Visualizing paths side by side...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Baseline path
    ax1.imshow(env.grid, cmap='binary', interpolation='nearest')
    if baseline_path:
        path_x, path_y = zip(*baseline_path)
        ax1.plot(path_x, path_y, 'b-', linewidth=2)
    ax1.plot(start[0], start[1], 'go', markersize=10)
    ax1.plot(goal[0], goal[1], 'ro', markersize=10)
    ax1.set_title(f"Baseline Path\nNodes Expanded: {len(baseline_expanded)}")
    
    # Constrained path
    ax2.imshow(env.grid, cmap='binary', interpolation='nearest')
    if constrained_path:
        path_x, path_y = zip(*constrained_path)
        ax2.plot(path_x, path_y, 'g-', linewidth=2)
    ax2.plot(start[0], start[1], 'go', markersize=10)
    ax2.plot(goal[0], goal[1], 'ro', markersize=10)
    ax2.set_title(f"Constrained Path\nNodes Expanded: {len(constrained_expanded)}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "paths", "paths_side_by_side.png"))
    plt.show()
    
    # Visualize node expansion
    print("Visualizing node expansion...")
    visualize_node_expansion(
        env,
        baseline_expanded,
        constrained_expanded,
        baseline_path,
        constrained_path,
        start,
        goal,
        save_path=os.path.join(log_dir, "visualizations", "node_expansion.png")
    )
    plt.show()  # Show node expansion visualization
    
    # Print statistics
    print("\nBaseline planner:")
    print(f"  Path length: {len(baseline_path)}")
    print(f"  Path smoothness: {baseline_smoothness:.4f}")
    print(f"  Nodes expanded: {len(baseline_expanded)}")
    print(f"  Search efficiency: {len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0:.4f}")
    print(f"  Constraint compliance: {baseline_compliance['overall']:.4f}")
    
    print("\nConstrained planner:")
    print(f"  Path length: {len(constrained_path)}")
    print(f"  Path smoothness: {constrained_smoothness:.4f}")
    print(f"  Nodes expanded: {len(constrained_expanded)}")
    print(f"  Search efficiency: {len(constrained_path) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0:.4f}")
    print(f"  Constraint compliance: {constrained_compliance['overall']:.4f}")
    
    # Calculate improvement metrics
    path_length_ratio = len(baseline_path) / len(constrained_path) if len(constrained_path) > 0 else 0
    nodes_expanded_ratio = len(baseline_expanded) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0
    search_efficiency_ratio = ((len(constrained_path) / len(constrained_expanded)) / 
                              (len(baseline_path) / len(baseline_expanded))) if len(baseline_expanded) > 0 and len(constrained_expanded) > 0 else 0
    compliance_improvement = constrained_compliance['overall'] - baseline_compliance['overall']
    
    print("\nImprovement:")
    print(f"  Path length ratio: {path_length_ratio:.2f}x")
    print(f"  Nodes expanded ratio: {nodes_expanded_ratio:.2f}x")
    print(f"  Search efficiency ratio: {search_efficiency_ratio:.2f}x")
    print(f"  Compliance improvement: {compliance_improvement:.4f}")
    
    # Save metrics
    metrics = {
        "constrained": {
            "success": True,
            "time": 0.0,
            "path_length": len(constrained_path),
            "path_smoothness": constrained_smoothness,
            "nodes_expanded": len(constrained_expanded),
            "search_efficiency": len(constrained_path) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0,
            "compliance": constrained_compliance
        },
        "baseline": {
            "success": True,
            "time": 0.0,
            "path_length": len(baseline_path),
            "path_smoothness": baseline_smoothness,
            "nodes_expanded": len(baseline_expanded),
            "search_efficiency": len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0,
            "compliance": baseline_compliance
        },
        "improvement": {
            "time_ratio": 0.0,
            "path_length_ratio": path_length_ratio,
            "nodes_expanded_ratio": nodes_expanded_ratio,
            "search_efficiency_ratio": search_efficiency_ratio,
            "compliance_improvement": compliance_improvement
        }
    }
    
    with open(os.path.join(log_dir, "metrics", "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nExperiment completed. Results saved to {log_dir}")

if __name__ == "__main__":
    run_efficiency_experiments()