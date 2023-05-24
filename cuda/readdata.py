import csv

output_file = "q2_5.txt"
csv_file = "q2_5results.csv"

# Open the output file for reading
with open(output_file, "r") as file:
    lines = file.readlines()

# Initialize the list to store the results
results = []

# Process the lines to extract the desired information
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Advection of"):
        # Extract Gx, Gy, Bx, By from the current line
        #_, _, _, _, _, _, _, Gx, Gy, _, _, _, _, _, Bx, By, _ = line.split()
        # print(lines[i+1].split())
        Gx, Gy = lines[i+1].split()[1].split("x")
        Bx, By = lines[i+1].split()[4].split("x")
        
        # Extract Advection time and GFLOPs rate from the next line
        #_, _, _, _, _, time, _, rate = lines[i + 2].split()
        # print(lines[i+2].split()[2], lines[i+2].split()[4])
        time = lines[i+2].split()[2][:-2]
        rate = lines[i+2].split()[4][5:]
        # Store the extracted information in a dictionary
        result = {
            "Gx": Gx,
            "Gy": Gy,
            "Bx": Bx,
            "By": By,
            "Advection time (sec)": time,
            "GFLOPs rate": rate
        }
        
        # Append the dictionary to the results list
        results.append(result)
    
    i += 1

# Write the results to the CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Gx", "Gy", "Bx", "By", "Advection time (sec)", "GFLOPs rate"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results have been stored in {csv_file}.")
