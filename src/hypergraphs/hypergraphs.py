# flake8: noqa
from abc import ABC

from manim import *


class OuterLabeledDot(VGroup, ABC):
    def __init__(self, position=ORIGIN, swatch=WHITE, label="", **kwargs):
        dot = Dot(position, color=swatch)
        super().__init__(dot, Tex(label).next_to(dot, 0.4 * UL), **kwargs)


class HypergraphExample(ThreeDScene):
    def construct(self):
        # Transition 1 - Introducing a hypergraph
        title = Tex(r"We can take a hypergraph")
        title.to_corner(UP + LEFT)
        d1 = (OuterLabeledDot([1, 2, 0], label="D1", swatch=YELLOW),)
        d2 = (OuterLabeledDot([-1, 1, 0], label="D2", swatch=BLUE),)
        d3 = (OuterLabeledDot([1.4, -1.4, 0], label="D3", swatch=GREEN),)
        d4 = (OuterLabeledDot([-3, -2, 0], label="D4", swatch=GREY),)
        d5 = OuterLabeledDot([3, 1, 0], label="D5", swatch=MAROON)

        disease = [d1[0], d2[0], d3[0], d4[0], d5]

        a1 = ArcPolygon([-4, -3, 1], [3, -2, 1], [-2, 2, 1])
        a2 = ArcPolygon([1, -2, 0.5], [4, 2, 1], [0, 3, 1], color=RED)

        edge1 = LabeledDot(Tex("e1", color=BLACK))
        edge2 = LabeledDot(Tex("e2", color=BLACK))

        self.play(
            Write(title),
        )

        self.wait()

        self.play(FadeIn(*disease))

        self.wait()

        edge1.next_to(a1)
        edge2.next_to(a2)

        self.play(Create(a1), Create(a2), FadeIn(edge1, edge2))

        self.wait()

        self.play(FadeOut(title, *disease, a1, a2))

        incidence_matrix = IntegerMatrix(
            [[0, 1], [1, 0], [1, 1], [1, 0], [0, 1]],
            left_bracket="(",
            right_bracket=")",
        )

        incidence_text = Tex(
            r"And convert it to an incidence matrix and a bipartite graph."
        )
        incidence_text.to_corner(UP + LEFT)

        self.play(
            FadeIn(incidence_text),
        )

        self.wait()

        self.add(edge2, incidence_matrix)

        self.wait()

        d = VGroup()
        d.add(*disease)

        l1 = Line(edge1, d[1])
        l2 = Line(edge1, d[2])
        l3 = Line(edge1, d[3])
        l4 = Line(edge2, d[2])
        l5 = Line(edge2, d[4])
        l6 = Line(edge2, d[0])

        self.play(FadeOut(incidence_text))

        self.wait()

        self.play(
            incidence_matrix.animate.shift(LEFT * 5.5, UP), FadeIn(*disease, a1, a2)
        )

        self.add(l1, l2, l3, l4, l5, l6)

        self.wait()

        self.play(FadeOut(a1, a2))

        self.wait()
