import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def inheritance(parent, genes_parent):
    """
    Resolves the probability of the child inheriting copies of the gene from one or both parent.
    """
    if parent == "mother" or "father":
        # The parent has 100% chance of transferring with two copies of the gene
        # The child will have the gene unless it mutates
        if genes_parent == 2:
            return 1 - PROBS["mutation"]
        # The parent has 50% chance of transferring with one copy of the gene
        elif genes_parent == 1:
            return 0.5
        # The parent has no gene
        # The child will not have the gene unless it mutates
        else:
            return PROBS["mutation"]
    else:
        raise Exception("invalid input")


def genes_num(person, one_gene, two_genes):
    """
    Resolves the number of gene of a person to 2, 1 or 0.
    """
    if person in two_genes:
        return 2
    elif person in one_gene:
        return 1
    else:
        return 0


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = 1

    for person in people:

        # Returns the number of genes the person has
        person_genes = genes_num(person, one_gene, two_genes)

        mother = people[person]["mother"]
        father = people[person]["father"]

        # where there is no parent, probability is calculated unconditionally
        if mother is None and father is None:
            probability *= PROBS["gene"][person_genes]

        # where there are parents, probability is calculated conditionally based on their genes
        else:
            inherit = {mother: 0, father: 0}

            genes_mother = genes_num(mother, one_gene, two_genes)
            genes_father = genes_num(father, one_gene, two_genes)

            inherit[mother] = inheritance(mother, genes_mother)
            inherit[father] = inheritance(father, genes_father)

            # Probability of the person having two copies of the gene from both parents
            if person_genes == 2:
                probability *= inherit[mother] * inherit[father]
            # Probability of the person having a copy of the gene from one of the parent
            elif person_genes == 1:
                probability *= inherit[mother] * (1 - inherit[father]) + (1 - inherit[mother]) * inherit[father]
            # Probability of the person not getting the gene from any of the parent
            else:
                probability *= (1 - inherit[mother]) * (1 - inherit[father])

        person_trait = person in have_trait
        # Return probability by multiplying by probability of the person having a trait
        probability *= PROBS["trait"][person_genes][person_trait]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        person_genes = genes_num(person, one_gene, two_genes)
        person_trait = person in have_trait

        # Updating "gene" distributions
        probabilities[person]["gene"][person_genes] += p
        # Updating "trait" distributions
        probabilities[person]["trait"][person_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        for field in probabilities[person]:
            sum_1 = sum(dict(probabilities[person][field]).values())
            for value in probabilities[person][field]:
                probabilities[person][field][value] /= sum_1


if __name__ == "__main__":
    main()
