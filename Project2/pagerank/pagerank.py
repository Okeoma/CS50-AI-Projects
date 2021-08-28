import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # If page has no outgoing links then generate randomly
    # the probability distribution of all pages with equal probability
    if not corpus[page]:
        # Initialise probability distribution to page chosen at random.
        prob_distribution = dict()
        # generate page rank for page
        for link in corpus:
            prob_distribution[link] = 1 / len(corpus)

    # if page has one of more outside links then generate probability distribution for each page.
    else:
        # Initialise probability distribution to Page with at least an outside link.
        prob_distribution = dict()
        for link in corpus:
            # generate the probabilities of pages with outside links from the corpus
            prob_distribution[link] = (1 - damping_factor) / len(corpus)
            # Add together the probabilities of all pages linked to the current page
            if link in corpus[page]:
                prob_distribution[link] += damping_factor / len(corpus[page])

    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialise page rank for all pages in dictionary to 0
    pagerank = dict(zip(corpus.keys(), [0] * len(corpus)))

    # Selecting a page at random for the first sample
    page = random.choice(list(corpus.keys()))

    # Using transition model, Sample repeatedly (steps, i - iteration to n)
    # for each page based on total pages in corpus including current page
    for i in range(n):
        sample_model = transition_model(corpus, page, damping_factor)
        distribution, link_weight = zip(*sample_model.items())
        page = random.choices(distribution, weights=link_weight, k=1)[0]

        # Next sample is generated from previous one based on it's transition model
        pagerank[page] += 1

    # Generating proportionate values for each page
    # in corpus by dividing counts by total no. of samples
    for page in corpus:
        pagerank[page] /= n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initializing PageRank before updating for the pages in corpus
    pagerank = dict()
    pagerank_updates = dict()

    # Assigning initial values of PageRank to pages in corpus
    for page in corpus:
        pagerank[page] = 1 / len(corpus)

    # Generating initial values for each page in corpus that will be used for iteration
    for page in corpus:
        pagerank_updates[page] = math.inf

    # Iteration of PageRank values until it reaches convergence threshold of 0.001
    while any(pagerank_update > 0.001 for pagerank_update in pagerank_updates.values()):
        for page in pagerank.keys():
            links_distribution = 0
            for p_connect, links in corpus.items():
                # Generating probability of pages with links for every page in corpus
                # By considering each linked page to the current page
                if page in links:
                    links_distribution += pagerank[p_connect] / len(links)

                # Generating probability of page without link in corpus
                # interpreted as having one link for each page in corpus including current page
                if not links:
                    links_distribution += pagerank[p_connect] / len(corpus)

            updated_pagerank = ((1 - damping_factor) / len(corpus)) + (damping_factor * links_distribution)

            # Iteratively tracking PageRank values until convergence with threshold
            pagerank_updates[page] = abs(updated_pagerank - pagerank[page])
            pagerank[page] = updated_pagerank

    return pagerank


if __name__ == "__main__":
    main()
