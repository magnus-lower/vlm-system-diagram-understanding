from methods.voting import get_items, normalize

COMPONENTS_PER_GROUP = 3


# This function identifies components first, and then it identifies connections in smaller groups
def run_ariadne(client, image_path, ask_model, load_text):
    components_prompt = load_text("prompts/two_stage_components.txt")
    components_result = ask_model(client, components_prompt, image_path)
    components = get_items(components_result, "Components")

    connections = []
    seen_connections = set()
    prompts = [components_prompt]

    for start in range(0, len(components), COMPONENTS_PER_GROUP):
        group = components[start:start + COMPONENTS_PER_GROUP]
        group_text = "\n".join(f"- {component}" for component in group)

        prompt = load_text("prompts/two_stage_connections.txt")
        prompt = prompt.replace("{components}", components_result)

        prompt = (
            "Identify the connections from these components:\n"
            f"{group_text}\n\n"
            "The components can also be connected to other components "
            "in the complete list.\n"
            "For connections with a direction, only include connections "
            "that start from these components.\n"
            "For connections without a direction, at least one of these "
            "components must be included.\n\n"
            + prompt
        )

        prompts.append(prompt)
        response = ask_model(client, prompt, image_path)
        normalized_group = {normalize(component, components) for component in group}

        for connection in get_items(response, "Connections"):
            if "->" in connection:
                source = connection.split("->", 1)[0]
                if normalize(source, components) not in normalized_group:
                    continue

            elif "||" in connection:
                left, right = connection.split("||", 1)
                if (normalize(left, components) not in normalized_group
                        and normalize(right, components) not in normalized_group):
                    continue

            else:
                continue

            normalized_connection = normalize(connection, components)

            if normalized_connection not in seen_connections:
                seen_connections.add(normalized_connection)
                connections.append(connection)

    result = (
        components_result.strip()
        + "\n\nConnections:\n"
        + "\n".join(f"- {component}" for component in connections)
    )

    return result, "\n\n".join(prompts)
