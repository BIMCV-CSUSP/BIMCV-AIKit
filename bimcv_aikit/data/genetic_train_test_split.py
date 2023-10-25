import numpy as np
import pandas as pd
import pygad


def metaheuristic_separation(dups, weigth):
    # dups
    desired_output = weigth * dups.sum()

    # desired_output_array = [w*dups.sum() for w in weights]
    def callback_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])
        print(f"Porcentaje particion de test : {np.abs((dups*ga_instance.best_solution()[0]).sum())/dups.sum()}")
        print(f"Porcentaje particion de train : {np.abs((dups*(1-ga_instance.best_solution()[0])).sum())/dups.sum()}")

    def fitness_func(ga_instance, solution, solution_idx):
        # solution = [2,0,3,1,...,1]
        # output_array = [np.array(dups)[(np.array(solution) == v).sum() for v in range(len(groups))]
        output = (dups * solution).sum()
        # fitness_class_array = (np.abs(output_array - desired_output)
        fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
        return fitness

    ga_instance = pygad.GA(
        num_generations=3000,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=40,
        num_genes=dups.shape[0],
        gene_type=int,
        init_range_low=0,
        init_range_high=2,
        parent_selection_type="sss",
        keep_parents=2,
        keep_elitism=0,
        crossover_type="single_point",
        mutation_type="swap",
        mutation_percent_genes=10,
        stop_criteria="reach_2.9",
        # callback_generation=callback_gen
    )
    ga_instance.run()
    return ga_instance.best_solution()


def separate_dataset(
    df,
    column_subjects,
    column_classes,
    new_column_name,
    label_partitions,
    label_percentages,
    shuffle=True,
    verbose=False,
):
    df_aux = df.copy().sample(frac=1).reset_index(drop=True) if shuffle else df.copy()
    df_aux[new_column_name] = label_partitions[-1]
    classes = np.array(df_aux[column_classes].unique())
    label_partitions_aux = label_partitions.copy()
    label_percentages_aux = label_percentages.copy()
    while len(label_partitions) > 1:
        label_2 = [label_partitions.pop(0), label_partitions[-1]]
        elem = label_percentages.pop(0)
        label_percentage_2 = np.array([elem, sum(label_percentages)]) / sum([elem, sum(label_percentages)])
        array_concats = []
        for class_ in classes:
            df_class = df_aux[(df_aux[column_classes] == class_) & (df_aux[new_column_name] == label_partitions[-1])]
            dups = df_class.pivot_table(index=[column_subjects, column_classes], aggfunc="size")
            solution, solution_fitness, solution_idx = metaheuristic_separation(dups, label_percentage_2[1])
            frame = {
                "weigths": dups,
                new_column_name: [label_2[sol] for sol in solution],
            }
            array_concats.append(pd.DataFrame(frame)[new_column_name])
        partitions = pd.concat(array_concats)
        df_aux[new_column_name] = df_aux.apply(
            (
                lambda row: partitions.loc[row[column_subjects], row[column_classes]]
                if (row[column_subjects], row[column_classes]) in partitions.index
                else row[new_column_name]
            ),
            axis=1,
        )
    if verbose:
        for class_ in classes:
            print(f"The images with class: {class_} are divided in:")
            for part, perc in zip(label_partitions_aux, label_percentages_aux):
                partition_count = df_aux[(df_aux[column_classes] == class_) & (df_aux[new_column_name] == part)][new_column_name].size
                print(
                    f"{part}({partition_count:5d}u.)-> {partition_count/df_aux[df_aux[column_classes]==class_].shape[0]*100:02.5f}% (expected {perc*100:02.5f}%)"
                )
    return df_aux
