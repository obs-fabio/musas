#!/bin/bash

python test/plot_curves.py --input_dir "./results/synthectic_experiments/depth/shallow" --output_dir ./results/plots/synthectic_experiments/depth
python test/plot_curves.py --input_dir "./results/synthectic_experiments/depth/deep" --output_dir ./results/plots/synthectic_experiments/depth
python test/plot_curves.py --input_dir "./results/synthectic_experiments/bottom/basalt" --output_dir ./results/plots/synthectic_experiments/bottom
python test/plot_curves.py --input_dir "./results/synthectic_experiments/bottom/clay" --output_dir ./results/plots/synthectic_experiments/bottom
python test/plot_curves.py --input_dir "./results/synthectic_experiments/bottom/gravel" --output_dir ./results/plots/synthectic_experiments/bottom
python test/plot_curves.py --input_dir "./results/synthectic_experiments/bottom/sand" --output_dir ./results/plots/synthectic_experiments/bottom
python test/plot_curves.py --input_dir "./results/synthectic_experiments/bottom/silt" --output_dir ./results/plots/synthectic_experiments/bottom
python test/plot_curves.py --input_dir "./results/synthectic_experiments/dynamic/fixed distance" --output_dir ./results/plots/synthectic_experiments/dynamic
python test/plot_curves.py --input_dir "./results/synthectic_experiments/dynamic/cpa in" --output_dir ./results/plots/synthectic_experiments/dynamic
python test/plot_curves.py --input_dir "./results/synthectic_experiments/dynamic/cpa out" --output_dir ./results/plots/synthectic_experiments/dynamic
python test/plot_curves.py --input_dir "./results/synthectic_experiments/season/summer" --output_dir ./results/plots/synthectic_experiments/season
python test/plot_curves.py --input_dir "./results/synthectic_experiments/season/autumn" --output_dir ./results/plots/synthectic_experiments/season
python test/plot_curves.py --input_dir "./results/synthectic_experiments/season/winter" --output_dir ./results/plots/synthectic_experiments/season
python test/plot_curves.py --input_dir "./results/synthectic_experiments/season/spring" --output_dir ./results/plots/synthectic_experiments/season
python test/plot_curves.py --input_dir "./results/synthectic_experiments/volume/reference" --output_dir ./results/plots/synthectic_experiments/volume
python test/plot_curves.py --input_dir "./results/synthectic_experiments/volume/non reference" --output_dir ./results/plots/synthectic_experiments/volume


python test/plot_curves.py --input_dir "./results/domain_experiment" --output_dir ./results/plots/iara
python test/plot_curves.py --input_dir "./results/domain_experiment" --output_dir ./results/plots/iemanja
python test/plot_curves.py --input_dir "./results/domain_experiment" --output_dir ./results/plots/combined
