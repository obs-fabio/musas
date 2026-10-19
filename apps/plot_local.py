import os
import lps_synthesis.database.scenario as syndb

def _main():
    base_dir = "./results/plots"
    os.makedirs(base_dir, exist_ok=True)
    syndb.Location.plot(os.path.join(base_dir, "locals.png"))

if __name__ == "__main__":
    _main()
