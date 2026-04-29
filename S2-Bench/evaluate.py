'''
For evaluation
'''
import argparse
import pandas as pd
from utils.evaluation import mol_prop, calculate_novelty, calculate_similarity
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="llama3.1-8B")

# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")
parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")

parser.add_argument("--output_dir", type=str, default="./new_predictions/")
parser.add_argument("--calc_novelty", action="store_true", default=False)

args = parser.parse_args()

raw_file = "./data/benchmarks/{}/{}/{}/test.csv".format(args.benchmark, args.task, args.subtask)
target_file = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/" + args.subtask + ".csv"

data = pd.read_csv(raw_file)
try:
    target = pd.read_csv(target_file)
except:
    target = pd.read_csv(target_file, engine='python')

# Limit datasets to first N samples if specified
if args.max_samples is not None and args.max_samples > 0:
    original_length = len(data)
    data = data.head(args.max_samples)
    target = target.head(args.max_samples)
    print(f"Evaluation limited to {len(data)} samples (original: {original_length})")
else:
    args.max_samples = len(data)

if args.benchmark == "open_generation":
    if args.task == "MolCustom":
        if args.subtask == "AtomNum":
            # accuracy
            atom_type = ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']
            flags = []
            valid_molecules = []
            
            # use tqdm to show the progress
            for idx in tqdm(range(len(data))):
                if mol_prop(target["outputs"].iloc[idx], "validity"):
                    valid_molecules.append(target["outputs"].iloc[idx])
                    flag = 1
                    for atom in atom_type:
                        if mol_prop(target["outputs"].iloc[idx], "num_" + atom) != int(data[atom][idx]):
                            flag = 0
                            break
                    flags.append(flag)
                else:
                    flags.append(0)
                # Novelty
                # novelty = mol_prop(target["outputs"].iloc[idx], "novelty")
                # if novelty is not None:
                #     novelties.append(novelty)
                
            
            
            
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))    
            if args.calc_novelty:
                print("Novelty: ", sum(novelties) / len(novelties))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Accuracy": [sum(flags) / len(flags)],
                "Validity": [len(valid_molecules) / len(flags)]
            }
            if args.calc_novelty:
                metrics_data["Novelty"] = [sum(novelties) / len(novelties)]
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(flags))),
                "generated_mol": target["outputs"].iloc[:len(flags)].tolist(),
                "SR": flags
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
                
                
        elif args.subtask == "FunctionalGroup":
            functional_groups = ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(len(data))):
                if mol_prop(target["outputs"].iloc[idx], "validity"):
                    valid_molecules.append(target["outputs"].iloc[idx])
                    flag = 1
                    for group in functional_groups:
                        if group == "benzene rings":
                            if mol_prop(target["outputs"].iloc[idx], "num_benzene_ring") != int(data[group][idx]):
                                flag = 0
                                break
                        else:
                            if mol_prop(target["outputs"].iloc[idx], "num_" + group) != int(data[group][idx]):
                                flag = 0
                                break
                    flags.append(flag)
                else:
                    flags.append(0)
                
                
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                print("Novelty: ", sum(novelties) / len(novelties))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Accuracy": [sum(flags) / len(flags)],
                "Validity": [len(valid_molecules) / len(flags)]
            }
            if args.calc_novelty:
                metrics_data["Novelty"] = [sum(novelties) / len(novelties)]
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(flags))),
                "generated_mol": target["outputs"].iloc[:len(flags)].tolist(),
                "SR": flags
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")

        elif args.subtask == "BondNum":
            bonds_type = ['single', 'double', 'triple', 'rotatable', 'aromatic']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(len(data))):
                if mol_prop(target["outputs"].iloc[idx], "validity"):
                    valid_molecules.append(target["outputs"].iloc[idx])
                    flag = 1
                    for bond in bonds_type:
                        if bond == "rotatable":
                            if int(data[bond][idx]) == 0:
                                continue
                            elif mol_prop(target["outputs"].iloc[idx], "rot_bonds") != int(data[bond][idx]):
                                flag = 0
                                break
                        else:
                            if int(data[bond][idx]) == 0:
                                continue
                            elif mol_prop(target["outputs"].iloc[idx], "num_" + bond + "_bonds") != int(data[bond][idx]):
                                flag = 0
                                break
                    flags.append(flag)
                else:
                    flags.append(0)
                
            
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                print("Novelty: ", sum(novelties) / len(novelties))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Accuracy": [sum(flags) / len(flags)],
                "Validity": [len(valid_molecules) / len(flags)]
            }
            if args.calc_novelty:
                metrics_data["Novelty"] = [sum(novelties) / len(novelties)]
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(flags))),
                "generated_mol": target["outputs"].iloc[:len(flags)].tolist(),
                "SR": flags
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")

    elif args.task == "MolEdit":
        if args.subtask == "AddComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                group = data["added_group"][idx]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"].iloc[idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) + 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
            
        elif args.subtask == "DelComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                group = data["removed_group"][idx]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"].iloc[idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) - 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
            
        elif args.subtask == "SubComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                added_group = data["added_group"][idx]
                removed_group = data["removed_group"][idx]
                if added_group == "benzene ring":
                    added_group = "benzene_ring"
                if removed_group == "benzene ring":
                    removed_group = "benzene_ring"

                target_mol = target["outputs"].iloc[idx]

                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + removed_group) == mol_prop(raw, "num_" + removed_group) - 1 and mol_prop(target_mol, "num_" + added_group) == mol_prop(raw, "num_" + added_group) + 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
            

    elif args.task == "MolOpt":
        if args.subtask == "LogP":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                target_mol = target["outputs"].iloc[idx]
                instruction = data["Instruction"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    if "lower" in instruction or "decrease" in instruction:
                        if mol_prop(target_mol, "logP") < mol_prop(raw, "logP"):
                            successed.append(1)
                        else:
                            successed.append(0)
                    else:
                        if mol_prop(target_mol, "logP") > mol_prop(raw, "logP"):
                            successed.append(1)
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")

        elif args.subtask == "MR":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                target_mol = target["outputs"].iloc[idx]
                instruction = data["Instruction"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    if "lower" in instruction or "decrease" in instruction:
                        if mol_prop(target_mol, "MR") < mol_prop(raw, "MR"):
                            successed.append(1)
                        else:
                            successed.append(0)
                    else:
                        if mol_prop(target_mol, "MR") > mol_prop(raw, "MR"):
                            successed.append(1)
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
            
        elif args.subtask == "QED":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                target_mol = target["outputs"].iloc[idx]
                instruction = data["Instruction"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    if "lower" in instruction or "decrease" in instruction:
                        if mol_prop(target_mol, "qed") < mol_prop(raw, "qed"):
                            successed.append(1)
                        else:
                            successed.append(0)
                    else:
                        if mol_prop(target_mol, "qed") > mol_prop(raw, "qed"):
                            successed.append(1)
                        else:
                            successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / len(data))
            
            # Save metrics to CSV
            metrics_file = target_file.replace(".csv", f"_metrics_N{args.max_samples}.csv")
            metrics_data = {
                "Success Rate": [sum(successed) / len(successed)],
                "Similarity": [sum(similarities) / len(similarities)],
                "Validity": [len(valid_molecules) / len(data)]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            for key, value in metrics_data.items():
                print(f"{key}: {value}")
            print(f"Metrics saved to {metrics_file}")
            
            # Save per-item results
            per_item_file = target_file.replace(".csv", f"_per_item_N{args.max_samples}.csv")
            per_item_df = pd.DataFrame({
                "idx": list(range(len(successed))),
                "generated_mol": target["outputs"].iloc[:len(successed)].tolist(),
                "SR": successed
            })
            per_item_df.to_csv(per_item_file, index=False)
            print(f"Per-item results saved to {per_item_file}")
            
elif args.benchmark == "targeted_generation":
    pass
else:
    raise ValueError("Invalid Benchmark Type")