import csv

def parse_output_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(len(lines)):
            line = lines[i].strip()

            if line.startswith("OMP_NUM_THREADS="):
                parameters = {}
                param_line = line.split()
                parameters["OMP_NUM_THREADS"] = int(param_line[0].split("=")[1])
                parameters["P"] = int(param_line[3])
                parameters["Advection_time"] = float(lines[i+2].split()[3].rstrip("s,"))
                results.append(parameters)

    return results

def write_to_csv(results, csv_file_path):
    fieldnames = ["OMP_NUM_THREADS", "P", "Advection_time"]
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Specify the path to your output file
output_file_path = "./q1_4.sh.o84918569"

# Specify the path to the output CSV file
csv_file_path = "./q1_4output.csv"

# Parse the output file
results = parse_output_file(output_file_path)

# Write the results to a CSV file
write_to_csv(results, csv_file_path)
