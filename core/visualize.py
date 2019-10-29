import sys
import sty
import tensorflow as tf
sty.fg.set_style('orange', sty.RgbFg(255, 100, 50))


def show_binary_rationale(ids, z, idx2word, tofile=False):
    """
    Visualize rationale.  
    Inputs:
        ids -- numpy of the text ids (sequence_length,).
        z -- binary rationale (sequence_length,).
        idx2word -- map id to word.
    """
    text = [idx2word[idx] for idx in ids]
    output = ""
    for i, word in enumerate(text):
        if z[i] == 1:
            output += sty.fg.orange + word + sty.fg.rs + " "
        else:
            output += word + " "

    if tofile:
        return output
    else:
        try:
            print(output)
        except Exception as e:
            print(e)

        sys.stdout.flush()


def show_binary_rationale_with_annotation(ids, z, r, idx2word, tofile=False):
    """
    Visualize rationale with factual rationale.  
    Inputs:
        ids -- numpy of the text ids (sequence_length,).
        z -- binary rationale (sequence_length,).
        r -- binary rationale annotation (sequence_length,).
        idx2word -- map id to word.
    """

    text = [idx2word[idx] for idx in ids]
    output = ""
    for i, word in enumerate(text):
        if z[i] == 1:
            output += sty.fg.blue

        if r[i] == 1:
            output += sty.ef.underl + word + sty.rs.underl
        else:
            output += word

        if z[i] == 1:
            output += sty.fg.rs

        output += " "

    if tofile:
        return output
    else:
        try:
            print(output)
        except Exception as e:
            print(e)

        sys.stdout.flush()

        sys.stdout.flush()
