![synonyms_picture](https://github.com/DonutMan06/DonutMan06/blob/main/littre.png)

# littre-analysis
Basic tools for Littre analysis (which is a French dictionary)

Some example of the functions of the Littre() class are given below:
* get_definition() : returns the definition of any given word
* print_longest/shortest_def() : prints a list of words for which the definition is the longest (the shortest). It appeared that the french words "prendre" ("take"), "main" ("hand") and "faire" ("do") have the longest definition according to the Littre
* print_most/least_narcissistic : prints the most (the least) narcissistics words of the dictionary. The narcissistic number of a given word is the number of times that word quotes itself in its definition
* plot_adjacency_matrix() : plots the adjacency matrix of the dictionary. In this structure, the nodes are the entries and the edges are defined by the definition sets.

Following Umberto's Eco literary essay "Opera aperta" (1962), I developped a tool that automatically assesses the poeticity of a given text.
Umberto Eco's main point of this essay is that poeticity comes from unlikelihood; which is strongly linked to Shannon's information theory.
Analysing the dictionary as a graph structure lets us compute a distance (in the mathematic meaning) between words.

I analyzed a poem from Breton & Soupault "Les Champs Magnétiques" (1920). Its poeticity matrix is plotted in the picture above : the lesser that number is, the better is the poeticity. But the main conclusion I've obtained so far is that the dictionary poorly links the words. The reason of that lies in these two facts :
* the Littre lists 75638 entries
* each entry is defined in 31 words (mean value)

That is to say that, according to the Littre, any given word has no relation with most of the other words.
A solution for this limitation might be to use iterates of the dictionary to arbitrarly densify the adjacency matrix. That is to say : making bigger sets by looking at the definitions of words included in the definition list of any entry.

## Usage

```python
import littre
x = littre.Littre()
```

## Improvements and known issues

* the poeticity matrix computation should be improved
* perhaps I should have split all the functions in the module instead of creating a big class


## Acknowledgment

The author wants to thank François Gannaz for having shared his XML Littre
under CC-by-SA 3.0 licence

Link : https://bitbucket.org/Mytskine/xmlittre-data/src/master/

Littré, Émile. Dictionnaire de la langue française. Paris, L. Hachette, 1873-1874.
Electronic version created by François Gannaz. http://www.littre.org
