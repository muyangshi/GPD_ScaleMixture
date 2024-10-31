"""
20241031
Create the Folder structure for coverage analysis, 
    and copy the required files into each simulation folder
Create bash scripts to submit simulate_data.py and sampler.py
Create master bash scripts to run the submissions

This file is placed in the GPD_ScaleMixture/script folder
"""

import os
import shutil

# Define paths and parameters
base_folder = "../" # the root GPD_ScaleMixture folder
script_folder = "./"  # Adjusted to current directory
coverage_folder = "coverage_analysis"
scenarios = ["scenario1", "scenario2", "scenario3"]
num_simulations = 5
files_to_copy = ["utilities.py", "RW_inte.py", "RW_inte_cpp.cpp", 
                 "simulate_data.py",
                 "proposal_cov.py", "sampler.py"]

# Create the base folder if it doesn't exist
os.makedirs(base_folder+coverage_folder, exist_ok=True)

for scenario in scenarios:
    # Create each scenario folder
    scenario_path = os.path.join(base_folder, coverage_folder, scenario)
    os.makedirs(scenario_path, exist_ok=True)
    
    # Create the scenario-level bash scripts
    scenario_run_simulate_script = os.path.join(scenario_path, "run_all_simulate.sh")
    scenario_run_sampler_script  = os.path.join(scenario_path, "run_all_sampler.sh")
    
    # Write the scenario-level script to run all simulations
    with open(scenario_run_simulate_script, "w") as run_simulate, open(scenario_run_sampler_script, "w") as run_sampler:
        run_simulate.write("#!/bin/bash\n")
        run_sampler.write("#!/bin/bash\n")
        
        for i in range(1, num_simulations + 1):
            simulation_folder = f"simulation_{i}"

            run_simulate.write(f"cd /projects/$USER/GPD_ScaleMixture/{coverage_folder}/{scenario}/{simulation_folder}\n")
            run_simulate.write(f"sbatch submit_simulate.sh\n")

            run_sampler.write(f"cd /projects/$USER/GPD_ScaleMixture/{coverage_folder}/{scenario}/{simulation_folder}\n")
            run_sampler.write(f"sbatch {simulation_folder}/submit_sampler.sh\n")
    
    for i in range(1, num_simulations + 1):
        # Create each simulation folder within the scenario
        simulation_folder_path = os.path.join(scenario_path, f"simulation_{i}")
        os.makedirs(simulation_folder_path, exist_ok=True)
        
        # Copy the required files into the simulation folder
        for file_name in files_to_copy:
            source_file = os.path.join(script_folder, file_name)
            destination_file = os.path.join(simulation_folder_path, file_name)
            shutil.copy2(source_file, destination_file)
        
        # Create bash scripts in each simulation folder to submit jobs to the cluster
        submit_simulate_script = os.path.join(simulation_folder_path, "submit_simulate.sh")
        submit_sampler_script = os.path.join(simulation_folder_path, "submit_sampler.sh")
        
        submit_simulate_script_lines = [
            "#!/bin/bash",
            "#SBATCH --account=csu70_alpine2",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=10",
            "#SBATCH --constraint=ib",
            "#SBATCH --partition=amilan",
            "#SBATCH --time=08:00:00",
            f"#SBATCH --job-name={scenario}_simulate_data_{i}",
            f"#SBATCH --output=OUTPUT-%x.%j.out",

            "module purge",
            "module load anaconda",
            "conda activate mcmc",
            "module load gcc",
            "module load gsl",
            "module load boost",

            "$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas",
            f"python3 simulate_data.py {i*13 + 7}"
        ]

        submit_sampler_script_lines = [
            "#!/bin/bash",
            "#SBATCH --account=csu70_alpine2",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=64",
            "#SBATCH --constraint=ib",
            "#SBATCH --partition=amilan",
            "#SBATCH --qos=long",
            "#SBATCH --time=7-00:00:00",
            f"#SBATCH --job-name={scenario}_sampler_{i}",
            f"#SBATCH --output=OUTPUT-%x.%j.out",

            "module purge",
            "module load anaconda",
            "conda activate mcmc",
            "module load gcc",
            "module load openmpi",
            "SLURM_EXPORT_ENV=ALL",
            "module load gsl",
            "module load boost",

            "$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC -L$CURC_GSL_LIB -L$CURC_BOOST_LIB RW_inte_cpp.cpp -shared -fPIC -o RW_inte_cpp.so -lgsl -lgslcblas",
            f"mpirun -n 64 python3 sampler.py {i*13 + 7}"
        ]

        # Write the script to submit simulate_data.py to the cluster
        with open(submit_simulate_script, "w") as simulate_script:
            simulate_script.write("\n".join(submit_simulate_script_lines))
        
        # Write the script to submit sampler.py to the cluster
        with open(submit_sampler_script, "w") as sampler_script:
            sampler_script.write("\n".join(submit_sampler_script_lines))

print("Folder structure, file copying, and script creation completed successfully!")