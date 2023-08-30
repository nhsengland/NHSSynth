from typing import Final, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network


class ConstraintGraph:
    _VALID_OPERATORS: Final = [">", ">=", "<", "<=", "in"]
    _POSITIVITY_TO_OPERATOR: Final = {"positive": ">", "nonnegative": ">=", "negative": "<", "nonpositive": "<="}
    _BRACKET_TO_OPERATOR: Final = {"[": ">=", "]": "<=", "(": ">", ")": "<"}
    _OPERATOR_TO_PANDAS: Final = {"<": pd.Series.lt, "<=": pd.Series.le, ">": pd.Series.gt, ">=": pd.Series.ge}

    class Constraint:
        def __init__(
            self,
            base: str,
            operator: str,
            reference: Union[str, float],
            reference_is_column: bool = False,
        ):
            self.base = base
            self.operator = operator
            self.reference = reference
            self.reference_is_column = reference_is_column

        def __str__(self) -> str:
            return f"{self.base} {self.operator} {self.reference}"

        def __repr__(self) -> str:
            return str(self)

        def __eq__(self, other) -> bool:
            return (
                self.base == other.base
                and self.operator == other.operator
                and self.reference == other.reference
                and self.reference_is_column == other.reference_is_column
            )

        def transform(self, df):
            base = df[self.base]
            if self.reference_is_column:
                reference = df[self.reference][base.index]
            else:
                reference = float(self.reference)

            adherence = self._OPERATOR_TO_PANDAS[self.operator](reference).astype(int)
            adherence[reference.isna()] = 1
            # When there is no reference, i.e. admidate and disdate, constraint is disdate >= admidate and admidate is null, we assume we want to keep the ability to generate disdates without a reference admidate, so we require a new column that inherits the constraints of the base column except for this constraint
            diff = abs(base[adherence] - reference[adherence])
            diff.fillna(diff.mean(), inplace=True)
            df[self.base + "_diff"] = np.log(diff + 1)
            df[self.base + "_adherence"] = adherence
            return df

    class ComboConstraint:
        def __init__(self, columns: list[str]):
            self.columns = columns

        def __str__(self) -> str:
            return f"fixcombo {' '.join(self.columns)}"

        def __repr__(self) -> str:
            return str(self)

        def __eq__(self, other) -> bool:
            return self.columns == other.columns

        def transform(self, df):
            return df

    def __init__(self, constraint_strings: Optional[list[str]], columns: pd.Index, metadata: dict):
        self._columns = columns
        self._metadata = metadata
        self.raw_constraint_strings = constraint_strings
        self.validated_constraint_strings = self.validate_constraint_strings()
        self.graph = self.build_graph(self.validated_constraint_strings)
        self.minimal_constraints = self.determine_minimal_constraints()
        self.minimal_graph = self.build_graph(
            [str(c).split(" ") for c in self.minimal_constraints if isinstance(c, self.Constraint)]
            + [
                ("fixcombo", str(c).split(" ")[1:])
                for c in self.minimal_constraints
                if isinstance(c, self.ComboConstraint)
            ]
        )

    def _validate_fixcombo_constraint(self, elements: list[str]) -> tuple[str, str]:
        for column in elements[1:]:
            self._column_exists(column)
            if not self._metadata[column].categorical:
                raise ValueError(f"'{column}' must be categorical to use the 'fixcombo' operator.")
        return ("fixcombo", elements[1:])

    def _column_exists(self, column: str) -> None:
        if not column in self._columns:
            raise ValueError(f"Constraint refers to a column that does not exist ('{column}').")

    def _validate_positivity(self, positivity: str) -> None:
        if positivity not in self._POSITIVITY_TO_OPERATOR:
            raise ValueError(f"Constraint has an invalid positivity specification ('{positivity}').")

    def _validate_simple_constraint(self, base: str, positivity: str) -> tuple[str, str, str]:
        self._column_exists(base)
        self._validate_positivity(positivity)
        return (base, self._POSITIVITY_TO_OPERATOR[positivity], "0")

    def _validate_operator(self, base: str, operator: str) -> None:
        if operator not in self._VALID_OPERATORS:
            raise ValueError(f"Constraint has an invalid operator ('{operator}').")
        if self._metadata[base].dtype.kind == "O":
            raise ValueError(
                f"Constraint's base column ('{base}') must be numeric or datetime when '{operator}' is used."
            )

    def _validate_matching_dtypes(self, base: str, reference: str) -> None:
        if self._metadata[base].dtype != self._metadata[reference].dtype:
            raise ValueError(
                f"Constraint's base column ('{base}') has a different dtype ({self._metadata[base].dtype.name}) to the reference column's ('{reference}': {self._metadata[reference].dtype.name}), which is not allowed."
            )
        if self._metadata[base].categorical or self._metadata[reference].categorical:
            raise ValueError(
                f"Constraint's base column ('{base}') and reference column ('{reference}') must both be numeric or datetime when using any operator other than 'fixcombo'."
            )

    def _get_range_operators(self, reference: str) -> tuple[str, str]:
        if reference[0] not in ["[", "("] or reference[-1] not in ["]", ")"]:
            raise ValueError(
                f"Constraint's reference is not a valid range specification ('{reference}'), it must be of the form '[' or '(' + 'a,b' + ']' or ')'."
            )
        return self._BRACKET_TO_OPERATOR[reference[0]], self._BRACKET_TO_OPERATOR[reference[-1]]

    def _validate_constant_dtype(self, base: str, reference: str) -> None:
        if self._metadata[base].dtype.kind == "O":
            raise ValueError(
                f"The reference ('{reference}') is not a valid dtype for the constraint's base column ('{base}': '{self._metadata[base].dtype}')."
            )
        elif self._metadata[base].dtype.kind == "M":
            try:
                pd.to_datetime(reference)
            except ValueError:
                raise ValueError(
                    f"The reference ('{reference}') is not a valid datetime to match the dtype of the constraint's base column ('{base}': '{self._metadata[base].dtype}')."
                )
        else:
            try:
                float(reference)
            except ValueError:
                raise ValueError(
                    f"The reference ('{reference}') is not a valid float to match the dtype of the constraint's base column ('{base}: '{self._metadata[base].dtype}')."
                )

    def _validate_range_component(self, base: str, component: str) -> str:
        component = component.strip()
        if component in self._columns:
            self._validate_matching_dtypes(base, component)
        else:
            self._validate_constant_dtype(base, component)
        return component

    def _validate_reference_constraint(self, base: str, operator: str, reference: str) -> list[tuple[str, str, str]]:
        self._column_exists(base)
        self._validate_operator(base, operator)
        if reference in self._columns:
            self._validate_matching_dtypes(base, reference)
        elif operator == "in":
            low_op, high_op = self._get_range_operators(reference)
            low, high = reference[1:-1].split(",")
            low = self._validate_range_component(base, low)
            high = self._validate_range_component(base, high)
            if low not in self._columns and high not in self._columns and float(low) >= float(high):
                raise ValueError(
                    f"Constraint's reference is not a valid range specification ('{reference}'), the lower bound must be strictly less than the upper bound."
                )
            return [(base, low_op, low), (base, high_op, high)]
        else:
            self._validate_constant_dtype(base, reference)
        return [(base, operator, reference)]

    def validate_constraint_strings(self) -> list[tuple[str, str, str]]:
        valid_constraints = []
        for constraint_string in self.raw_constraint_strings:
            elements = constraint_string.split(" ")
            if elements[0] == "fixcombo":
                valid_constraints.append(self._validate_fixcombo_constraint(elements))
            elif len(elements) == 2:
                valid_constraints.append(self._validate_simple_constraint(*elements))
            elif len(elements) == 3:
                valid_constraints.extend(self._validate_reference_constraint(*elements))
            else:
                raise ValueError(f"Constraint '{constraint_string}' is invalid.")
        return valid_constraints

    def build_graph(self, constraint_string_tuples) -> nx.DiGraph:
        graph = nx.DiGraph()
        for col in self._columns:
            graph.add_node(col, color="purple" if self._metadata[col].categorical else "blue")
        for cst in constraint_string_tuples:
            if cst[0] == "fixcombo":
                cols = cst[1]
                for i in range(len(cols) - 1):
                    graph.add_edge(cols[i], cols[i + 1], color="purple")
            else:
                item1, operator, item2 = cst
                if "<" in operator:
                    item1, item2 = item2, item1
                if item1 not in graph.nodes:
                    graph.add_node(item1)
                    graph.nodes[item1]["color"] = "red"
                if item2 not in graph.nodes:
                    graph.add_node(item2)
                    graph.nodes[item2]["color"] = "red"
                graph.add_edge(item1, item2, color="black" if len(operator) == 1 else "green")
                graph.add_edge(item1, item2, color="black" if len(operator) == 1 else "green")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                f"Constraint graph is not acyclic as required; some constraints involving {[c for c in nx.simple_cycles(graph)]} are invalid."
            )
        return graph

    def _check_constants_are_monotonic(self, longest_path: list[str], subgraph: nx.DiGraph) -> None:
        prev = None
        for node in longest_path:
            if subgraph.nodes[node]["color"] == "red":
                try:
                    new = float(node)
                except:
                    new = pd.to_datetime(node)
                if prev is not None and prev < new:
                    raise ValueError(
                        f"The constraints are inconsistent, '{prev}' is less than '{new}' but the sequence of constants in a chain of constraints must be monotonically increasing."
                    )
                prev = new

    def _traverse_longest_path(
        self,
        longest_path: list[str],
        subgraph: nx.DiGraph,
        constraints: list[Union[Constraint, ComboConstraint]],
    ) -> None:
        self._check_constants_are_monotonic(longest_path, subgraph)
        for i in range(len(longest_path) - 1):
            item1, item2 = longest_path[i], longest_path[i + 1]
            ref_is_col, operator = True, ">"
            if subgraph.edges[item1, item2]["color"] == "green":
                operator += "="
            if subgraph.nodes[item1]["color"] == "red":
                item1, item2 = item2, item1
                ref_is_col, operator = False, operator.replace(">", "<")
            constraint = self.Constraint(item1, operator, item2, reference_is_column=ref_is_col)
            if constraint not in constraints:
                constraints.append(constraint)
        return constraints

    def _determine_minimal_subgraph_constraints(
        self,
        subgraph: nx.DiGraph,
        constraints: list[Union[Constraint, ComboConstraint]],
    ) -> None:
        sources = [n for n in subgraph.nodes if subgraph.out_degree(n) == 0 and subgraph.in_degree(n) > 0]
        sinks = [n for n in subgraph.nodes if subgraph.in_degree(n) == 0 and subgraph.out_degree(n) > 0]
        for source in sources:
            for sink in sinks:
                paths = [p for p in nx.all_simple_paths(subgraph, sink, source)]
                paths_nodes = {n for p in paths for n in p}
                while paths_nodes:
                    longest_path = max(paths, key=len)
                    paths_nodes -= set(longest_path)
                    paths = [p for p in paths if not set(p).issubset(set(longest_path))]
                    constraints = self._traverse_longest_path(longest_path, subgraph, constraints)
        return constraints

    def determine_minimal_constraints(self) -> list[Constraint]:
        combo_constraints = []
        constraints = []
        all_subgraphs = [self.graph.subgraph(g) for g in nx.weakly_connected_components(self.graph) if len(g) > 1]
        for subgraph in all_subgraphs:
            if all([subgraph.edges[e]["color"] == "purple" for e in subgraph.edges]):
                combo_constraints.append(self.ComboConstraint(subgraph.nodes))
            else:
                constraints = self._determine_minimal_subgraph_constraints(subgraph, constraints)
        return combo_constraints + constraints

    def _output_graphs_html(self, name: str) -> None:
        if not hasattr(self, "graph") or not hasattr(self, "minimal_graph"):
            raise ValueError("Constraint graphs have not been built yet.")
        net = Network(directed=True, notebook=False, height="100%", width="100%")
        net.from_nx(self.graph)
        html = net.generate_html(notebook=False)
        with open(name, "w") as f:
            f.write(html)
        net = Network(directed=True, notebook=False, height="100%", width="100%")
        net.from_nx(self.minimal_graph)
        html = net.generate_html(notebook=False)
        with open(str(name).replace(".html", "_minimal.html"), "w") as f:
            f.write(html)
