from cat.mad_hatter.decorators import hook, tool
import json
from numpy import dot
from numpy.linalg import norm


@hook
def agent_prompt_prefix(prefix, cat):

    prefix = """
    Sei un farmacista, e rispondi in modo professionale.
    Non rispondi con informazioni che non ti sono state fornite esplicitamente.
    Non rispondi a domande inappropriate.
    """

    return prefix


filenames = []
filename_ids = {}


@hook
def before_rabbithole_insert_memory(doc, cat):
    global filenames
    global filename_ids

    if doc.metadata["source"] not in filenames:
        # doc.page_content
        name_f_string = (
            str(
                cat.llm(
                    f"""Rispondi SOLAMENTE con un file json formattato in questo modo:
                {{"name": qua scrivi il nome del farmaco indicato, solo il nome, senza indicare a cosa serve}}

                Dati : {doc.page_content}
        """
                )
            )
            .replace("```json", "")
            .replace("```", "")
        )
        name_f = json.loads(name_f_string)["name"]

        filenames.append(doc.metadata["source"])
        filename_ids.update({doc.metadata["source"]: name_f})

    doc.metadata["name"] = filename_ids[doc.metadata["source"]]

    return doc


def names_from_metadata(cat):
    points = cat.memory.vectors.collections["declarative"].get_all_points()
    name_list = []

    # Record -> id; payload={page_content; metadata}; vector

    for point in points:
        metadata_name = point.payload["metadata"]["name"]

        if metadata_name not in name_list:
            name_list.append(metadata_name)

    return name_list


def get_query_medicine(cat, names_list):

    user_message = cat.working_memory.user_message_json.text

    list_str = ""

    for i, name in enumerate(names_list):
        list_str += f"{i}). {name} \n"

    query_medicine_json = (
        cat.llm(
            f"""
    Your task is to produce a json containing the index of the medicine in the list which the user is talking about:

    User message {user_message}

    list of medicines {list_str}

    JSON:
    {{"index": // type int, if you can't find it in the list it's -1 }}
    """
        )
        .replace("```json", "")
        .replace("```", "")
    )

    index = json.loads(query_medicine_json)["index"]

    return names_list[index] if index != -1 else None


def content_and_vector_from_points(points):
    return_list = []

    for point in points:
        return_list.append((point.payload["page_content"], point.vector))

    return return_list


def cosine_similarity(memory_vector, query_vector):
    return dot(memory_vector, query_vector) / norm(memory_vector) * norm(query_vector)


def filter_points_by_name_meta(points, query_medicine):  # returns the tuple list
    # point -> ("str",[vett])

    filtered_points = [
        point for point in points if point.payload["metadata"]["name"] == query_medicine
    ]
    return [
        (
            p.payload["page_content"],
            p.vector,
        )
        for p in filtered_points
    ]


def get_three_best_points(
    query_medicine, tuple_list, cat
):  # Tuple list ---> [ (page_content, vector,)  ,]
    point_of_name = cat.embedder.embed_query(query_medicine)

    similarity_list = []

    for t in tuple_list:
        t_similarity = cosine_similarity(t[1], point_of_name)
        similarity_list.append((t[0], t[1], t_similarity))

    similarity_list_sorted = sorted(similarity_list, key=lambda x: x[2])
    similarity_list_sorted.reverse()

    # best_points = []

    # for _ in range(3):
    #     index = similarity_list.index(max(similarity_list))
    #     similarity_list.pop(index)
    #     best_points.append(
    #         (
    #             tuple_list[index][0],
    #             tuple_list[index][1],
    #             similarity_list[index],
    #         )
    #     )

    return similarity_list_sorted[:3]


@hook
def after_cat_recalls_memories(cat):
    names_list = names_from_metadata(cat)
    name_selected_user = get_query_medicine(cat, names_list)

    points = cat.memory.vectors.collections["declarative"].get_all_points()
    # vector_tuples = content_and_vector_from_points(points)

    three_best_results = []
    if name_selected_user:
        tuples = filter_points_by_name_meta(points, name_selected_user)

        three_best_results = get_three_best_points(
            cat.working_memory.user_message_json.text, tuples, cat
        )
    else:
        query_unknown = f"Non è presente nessuna informazione a riguardo di {cat.working_memory.user_message_json.text}"
        point_unknown = cat.embedder.embed_query(query_unknown)

        three_best_results = [(query_unknown, point_unknown, 0) for _ in range(3)]

    # Record -> id; payload={page_content; metadata}; vector
    # -> [3* (page_content,vector,similarity)]

    while len(three_best_results) < 3:
        three_best_results.append(three_best_results[0])

    for i in range(3):
        dec_memories = cat.working_memory.declarative_memories[i]

        new_document = dec_memories[0]
        new_document.page_content = three_best_results[i][0]
        new_document.metadata["source"] = name_selected_user

        dec_memories_new = (
            new_document,
            three_best_results[i][2],
            three_best_results[i][1],
            dec_memories[3],
        )

        cat.working_memory.declarative_memories[i] = dec_memories_new


@tool(return_direct=True)
def known(tool_input, cat):
    """Reply only to the question "How many medicines you know?" or to others which are stricly similar.
    Ignore generic questions about medicines.
    Input is always None"""

    names_list = names_from_metadata(cat)

    out_text = ""
    for number, name in enumerate(names_list):
        out_text += f"{number+1}. {name}\n"

    return out_text

# @tool(return_direct=True)
# def known(tool_input, cat):
#     """Reply only to the question 'Medicina'"""

#     # Test

#     return "ciao"
