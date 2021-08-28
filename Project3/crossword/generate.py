import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Initialise empty set for wrong words that are node consistent by looping through each variable
        for var in self.domains:
            wrong_word = set()

            # Checking for node consistency by looping through the words in the variable's domain
            for word in self.domains[var]:
                if len(word) != var.length:
                    wrong_word.add(word)

            # Remove words that are not node consistent from variable's domain
            for word in wrong_word:
                self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False

        # Select overlapping values (characters) of words that intersect to confirm if they are arc consistent.
        overlap = self.crossword.overlaps[x, y]
        i = overlap[0]
        j = overlap[1]

        for x_domain in self.domains[x]:
            remove = True

            for y_domain in self.domains[y]:
                if x_domain[i] == y_domain[j]:
                    remove = False

            # removing values from x-domain for which there is no corresponding value for y in y-domain.
            if remove:
                self.domains[x].remove(x_domain)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        arc_list = list()
        if arcs is None:
            for x in self.domains.keys():
                for y in self.domains.keys():
                    if x != y:
                        arc_list.append((x, y))
        else:
            for arc in arcs:
                arc_list.append(arc)

        while arcs:
            arc = arc_list.pop()
            if arc is not None:
                x = arc[0]
                y = arc[1]
                if self.revise(x, y):
                    if len(self.domains[x]) == 0:
                        return False
                    for z in self.crossword.neighbors(x) - self.domains[y]:
                        arc_list.append((z, x))
            else:
                break
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        if all(assignment.values()) and self.crossword.variables == assignment.keys():
            return True
        else:
            return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        check_val = set()

        for val in assignment:

            # Check that all values must be unique
            if val in check_val:
                return False
            else:
                check_val.add(assignment[val])

            # Check that there are no conflicting neighbors
            for neighbor in self.crossword.neighbors(val):
                if neighbor in assignment:
                    check_neighbor = self.crossword.overlaps[val, neighbor]
                    i = check_neighbor[0]
                    j = check_neighbor[1]
                    if assignment[val][i] != assignment[neighbor][j]:
                        return False

            # Confirming the length of each word have correct length
            if val.length != len(assignment[val]):
                return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        values = dict()
        neighbors = self.crossword.neighbors(var)
        for var in self.domains[var]:
            if var in assignment:
                continue
            else:
                values[var] = 0
                for neighbor in neighbors:
                    if var in self.domains[neighbor]:
                        values[var] += 1
        return sorted(values, key=lambda key: values[key])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Start with an unassigned variable
        un_var = self.crossword.variables - assignment.keys()

        # Confirming the amount of values remaining in the variable domain
        rem_val = {var: len(self.domains[var]) for var in un_var}
        sort_val = sorted(rem_val.items(), key=lambda x: x[1])

        # Returns variable with the minimum amount of values remaining in the domain if there is a tie
        if len(sort_val) == 1 or sort_val[0][1] != sort_val[1][1]:
            return sort_val[0][0]

        # Returns variable with the largest degree amongst neighbors If there is a tie
        else:
            deg = {var: len(self.crossword.neighbors(var)) for var in un_var}
            sort_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)
            return sort_deg[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Check if assignment is complete
        if self.assignment_complete(assignment):
            return assignment

        # Start with an unassigned variable
        var = self.select_unassigned_variable(assignment)

        # Loop through variable domains and assign correct value to variable
        for val in self.domains[var]:
            assignment[var] = val

            # If value is consistent with assignment, call backtracking search function
            if self.consistent(assignment):
                result = self.backtrack(assignment)

                # If result did not fail, return the result
                if result is not None:
                    return result

            # Remove value from the variable
            assignment.pop(var)

        # If no assignment is possible, return None
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()