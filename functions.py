import json
from typing import List, Optional

from cat.looking_glass.cheshire_cat import CheshireCat


def names_from_metadata(cat: CheshireCat) -> List[str]:
    """Use this function to get all the medicine names from the metadata in the vector database"""

    points = cat.memory.vectors.collections["declarative"].get_all_points()
    name_list = []

    # Record -> id; payload={page_content; metadata}; vector

    for point in points:
        metadata_name = point.payload["metadata"]["name"]

        if metadata_name not in name_list:
            name_list.append(metadata_name)

    return name_list


def get_query_medicine(
    cat: CheshireCat, names_list: List[str], user_message: str
) -> Optional[str]:
    list_str = ""

    for i, name in enumerate(names_list):
        list_str += f"{i}). {name} \n"

    query_medicine_json = (
        cat.llm(
            f"""
    Your task is to produce a json containing the index of the medicine in the list which the user is talking about:

    {{User message}} {user_message}

    {{list of medicines}} {list_str}

    JSON:
    {{"index": // type int, if you can't find it in the list or the user message does not contain the name it's -1 }}
    """
        )
        .replace("```json", "")
        .replace("```", "")
    )

    index = json.loads(query_medicine_json)["index"]

    return names_list[index] if index != -1 else None
