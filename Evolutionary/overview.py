import random


def create_population(size):
    """ creates a random population by the size of size. """


def fitness(specimen):
    """ Get fitness of specimen. """


def grade(populi):
    """ Gets average fitness of populi. """


def evolve(populi, retain, random_select, mutate):
    """ Evolves the populi by the best performing ones with configurable randomness. """


if __name__ == '__main__':
    # Gekozen voor opdracht 6.2
    random.seed(0)

    populi = create_population(size=10_000)
    fitness_history = [grade(populi)]

    print("start", fitness_history[-1])

    for _ in range(25):
        populi = evolve(populi=populi, retain=0.5, random_select=0.0, mutate=0.5)
        score = grade(populi)
        fitness_history.append(score)
        #print(f"{score} is de gemiddelde score.")

    print("end", fitness_history[-1])

