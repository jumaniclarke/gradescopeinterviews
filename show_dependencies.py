import spacy
from spacy import displacy

# Load spaCy model
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        from spacy.cli import download
        download('en_core_web_lg')
        nlp = spacy.load('en_core_web_lg')
    return nlp

def _print_dependency_table(doc):
    rows = []
    for token in doc:
        if token.is_space:
            continue
        rows.append(
            (
                str(token.i),
                token.text,
                token.lemma_,
                token.pos_,
                token.tag_,
                token.dep_,
                token.head.text,
                str(token.head.i),
            )
        )

    headers = ("i", "text", "lemma", "pos", "tag", "dep", "head", "head_i")
    all_rows = [headers, *rows]
    col_widths = [max(len(r[c]) for r in all_rows) for c in range(len(headers))]

    def fmt(row):
        return "  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt(row))


def show_dependencies(text, *, svg_path='dependency_parse.svg', print_table=True):
    nlp = load_spacy_model()
    doc = nlp(text)

    if print_table:
        _print_dependency_table(doc)

    # Render dependency parse as SVG
    svg = displacy.render(doc, style='dep', jupyter=False)
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f'Dependency diagram saved as {svg_path}')

if __name__ == '__main__':
    # Enter your text here
    #text = "I'm not sure currently, but in the past there was obviously internet issues, but now I'm not sure about internet issues."
    #text = "And sometimes if the pencil, obviously if the pencil is too light, even in sketches, then you, well, I do say it in class, make sure it's dark pencil that you use, but the pen creeps up. Unreadable scans, yes, when they use pencils"
    #text = "But I will delete the recordings once you've got the transcriptions"
    #text = "You know in the unfortunate event of the programme getting hacked and also network issues so that I think is a disadvantage or something I'd be concerned about when it comes to digital tools. "
    #text = "And also the fact that it's not, whats the word equitable, equitable for all students."
    #text= "Also it also saves time because if you want to maybe go back to a previous question, you just have to click and in an instant in a second it's available to you, whereas if you maybe wanted to, if you have a physical script you have to scroll back to like, you have to physically turn pages which can take time. "
    #text = "I think some of the challenges are that sometimes if it's not set up correctly and it's not, so if the back end works, so the setting up of the quizzes and making sure that the correct answers are available. "
    #text = "So the problem is that you have to manually group the answers which is time for consuming because you obviously want to use the tool for that function but the function is not working as properly as it should. "
    text = "Grade scope, not so much."
    show_dependencies(text)
