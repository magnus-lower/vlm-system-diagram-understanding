
from collections import Counter

MIN_VOTES = 3


# This function cleans the component names in order to make them easier to compare
def clean_name(name):
    name = " ".join(name.lower().split())
    name = name.replace(", ", ",")

    if name.startswith("unlabeled component"):
        name = name.split("(")[0].strip()

    return name


# This function matches a name with the identified components
def normalize_name(name, components):
    name = clean_name(name)

    for component in sorted(components, key=len, reverse=True):
        component = clean_name(component)

        if name == component or name.startswith(component + " ("):
            return component

    return name


# This function normalizes components and connections in order to make them easier to compare
def normalize(item, components):
    if "->" in item:
        left, right = item.split("->", 1)

        left = normalize_name(left, components)
        right = normalize_name(right, components)

        return left + " -> " + right

    if "||" in item:
        left, right = item.split("||", 1)

        names = sorted([
            normalize_name(left, components),
            normalize_name(right, components)
        ])

        return " || ".join(names)

    return normalize_name(item, components)


# This function extracts components or connections from the model's response
def get_items(result, section):
    header = section + ":"

    if header not in result:
        raise ValueError("Missing " + header)

    text = result.split(header, 1)[1]

    if section == "Components":
        text = text.split("Connections:", 1)[0]

    items = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("-"):
            items.append(line[1:].strip())

    return items


# This function keeps the items that appear in at least three runs
def majority_vote(runs, component_runs):
    votes = Counter()
    originals = {}

    for run, components in zip(runs, component_runs):
        seen = set()

        for item in run:
            key = normalize(item, components)

            if key not in seen:
                votes[key] += 1
                originals.setdefault(key, item)
                seen.add(key)

    return [
        originals[key]
        for key in votes
        if votes[key] >= MIN_VOTES
    ]


# Then, this function combines the responses from the five runs using majority voting
def vote_results(results):
    component_runs = []
    connection_runs = []

    for result in results:
        component_runs.append(get_items(result, "Components"))
        connection_runs.append(get_items(result, "Connections"))

    components = majority_vote(component_runs, component_runs)
    connections = majority_vote(connection_runs, component_runs)

    output = "Components:\n"

    for component in components:
        output += f"- {component}\n"

    output += "\nConnections:\n"

    for connection in connections:
        output += f"- {connection}\n"

    return output
