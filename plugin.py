from cat.mad_hatter.decorators import hook, tool
from cat.plugins.CC_plugin_foglietti_illustrativi.functions import *
import json


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


@hook
def before_cat_recalls_declarative_memories(declarative_recall_config, cat):
    medicine = "Tachipirina"
    # history = cat.working_memory.history
    # max_index = len(history) - 1
    # index_now = max_index

    # while medicine == -1:
    #     medicine = get_query_medicine(cat, names_from_metadata(cat), history[index_now])

    #     index_now -= 1
    #     if index_now < 0:
    #         medicine = ""

    declarative_recall_config["metadata"] = {"name": medicine}

    return declarative_recall_config


@tool(return_direct=True)
def how_many_medicines_known(tool_input, cat):
    """Reply only to the question "How many medicines you know?" or to others which are stricly similar.
    Ignore generic questions about medicines.
    Input is always None"""

    names_list = names_from_metadata(cat)

    out_text = ""
    for number, name in enumerate(names_list):
        out_text += f"{number+1}. {name}\n"

    return out_text