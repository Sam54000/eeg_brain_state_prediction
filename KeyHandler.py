import copy


# %%
class KeyHandler:
    def __init__(self) -> None:
        self.method_flag = None
        self.subjects = list()
        self.sessions = list()
        self.runs = list()
        self.tasks = list()
        self.descriptions = list()

    def transform_format(self):
        pass

    def read_value_format(self, value: str):
        prefix = "".join(filter(lambda x: x.isalpha(), value))
        pass

    def extract_key_value_pairs(self, key_value_string: str):
        key_value_pair_dict = dict()
        possible_keys = [
            "subject",
            "session",
            "task",
            "run",
            "description",
        ]
        if isinstance(key_value_string, str):
            separated_pairs = key_value_string.split("_")

        for pair in separated_pairs:
            key, value = pair.split("-")
            for possible_key in possible_keys:
                if key in possible_key:
                    key_value_pair_dict[possible_key] = value

        return key_value_pair_dict

    def read_keys_list(self, key_list: list):
        if self.method_flag is not None:
            raise Exception(
                "You already called a method on this object. To prevent overwriting parameters, re-initiate another object."
            )
        self.method_flag = "read"
        for keys in key_list:
            key_value_pair_dict = self.extract_key_value_pairs(keys)
            for key, value in key_value_pair_dict.items():
                attribute_name = key + "s"
                getattr(self, attribute_name).append(value)

    def select_from_list(
        self,
        key_list: list,
        subjects: list | str | None = None,
        sessions: list | str | None = None,
        tasks: list | str | None = None,
        descriptions: list | str | None = None,
        runs: list | str | None = None,
        how: str = "include",
    ):
        """Select a subset of data names based on given criteria."""

        def match_criteria(keys, criteria, key_type):
            if criteria is None:
                return True
            key_value = self.extract_key_value_pairs(keys).get(key_type)
            if isinstance(criteria, str):
                criteria = [criteria]
            if how == "include":
                return key_value in criteria
            elif how == "exclude":
                return key_value not in criteria
            return False

        selected_keys = []
        for keys in key_list:
            if (
                match_criteria(keys, subjects, "subject")
                and match_criteria(keys, sessions, "session")
                and match_criteria(keys, tasks, "task")
                and match_criteria(keys, descriptions, "description")
                and match_criteria(keys, runs, "run")
            ):
                selected_keys.append(keys)

        return selected_keys

    def generate_one_key_list(
        self,
        key_name: str,
        prefix: str | None,
        start: int | None,
        stop: int | None,
        nb_digit: int,
    ):
        if self.method_flag is not None:
            raise Exception(
                "You already called a method on this object. To prevent overwriting parameters, re-initiate another object."
            )
        self.method_flag = "write"

        for i in range(start, stop):
            string_index = str(i).rjust(nb_digit, "0")
            key_string = prefix + string_index
            getattr(self, key_name).append(key_string)

        return self

    def generate_all_keys(
        self, subject_parameters: dict, session_parameters: dict, tasks_parameters: dict
    ):
        if self.method_flag is not None:
            raise Exception(
                "You already called a method on this object. To prevent overwriting parameters, re-initiate another object."
            )
        self.method_flag = "write"

        subjects = [
            f'sub-{str(i).rjust(subject_parameters["nb_digit"], "0")}'
            for i in range(subject_parameters["start"], subject_parameters["stop"])
        ]
        sessions = [
            f'ses-{str(i).rjust(session_parameters["nb_digit"], "0")}'
            for i in range(session_parameters["start"], session_parameters["stop"])
        ]
        tasks = [f"task-{task}" for task in tasks_parameters["names"]]

        key_list = []
        for subject in subjects:
            for session in sessions:
                for task in tasks:
                    for run in range(1, tasks_parameters["runs"] + 1):
                        key_list.append(
                            f'{subject}_{session}_{task}_run-{str(run).rjust(2, "0")}'
                        )

        return key_list


# %%


class DataNumComponent:
    def __init__(self) -> None:
        self.prefix = None
        self.nb_digit = 3
        self.instance_type = "numerical"

    def __repr__(self):
        string_repr = []
        string_repr.append(f"{self.component_type} DataComponent instance")
        if self.prefix:
            string_repr.append(f"preffix: {self.prefix}")
        if getattr(self, "_values", False):
            string_repr.append(f"values: {self._values}")
        string_repr.append(f"nb_digits: {self.nb_digit}")
        if getattr(self, "formated_strings", False):
            string_repr.append(f"generated list: {self.formated_strings}")

        return "\n".join(string_repr)

    def __iter__(self):
        pass

    def _generate_values(self, start: int, stop: int) -> "DataNumComponent":
        """Generate a list of integers.

        Args:
            start (int): Integer to start from
            stop (int): Integer to end (included)

        Returns:
            DataNumComponent: The instance object
        """
        self._values = list()
        for value in range(start, stop):
            self._values.append(value)

        return self

    def _format_type(self) -> "DataNumComponent":
        if self.component_type.lower() == "task":
            self.key = self.component_type
        else:
            self.key = self.component_type[:3]
        self.component_type = self.component_type

        return self

    def copy(self) -> "DataNumComponent":
        """Perform a copy of the instance.

        Returns:
            DataNumComponent: The instance copied
        """
        return copy.deepcopy(self)

    def _extract_digits(self, values: int | str | list[int] | list[str]) -> list:
        """Private method to extract the digits from a various possibilities.

        Args:
            values (int | str | list[int] | list[str]): The values to extract
                                                        digits from

        Returns:
            DataNumComponent: The instance object
        """
        digits = []
        if isinstance(values, str):
            digits.append(int("".join([s for s in values if s.isdigit()])))

        elif isinstance(values, list):
            for value in values:
                if isinstance(value, str):
                    digits.append(int("".join([s for s in value if s.isdigit()])))
                elif isinstance(value, int):
                    digits.append(value)

        return digits

    def pick(self, to_pick: int | str | list[int] | list[str]) -> "DataNumComponent":
        """Select a subset.

        Args:
            to_pick (int | str | list[int] | list[str]): The selection to pick.

        Returns:
            DataNumComponent: The modified instance.
        """
        if getattr(self, "_values", False):
            vals = set(self._values)
            digits = set(self._extract_digits(to_pick))
            self._values = list(vals.intersection(digits))
        else:
            self._values = self._extract_digits(to_pick)
        self._format_string_list()

        return self

    def exclude(
        self, to_exclude: int | str | list[str] | list[str]
    ) -> "DataNumComponent":
        """Exclude the desired values.

        Args:
            to_exclude (int | str | list[str] | list[str]): The value to exclude

        Returns:
            DataComponent: The modified instance.
        """
        if getattr(self, "_values", False):
            vals = set(self._values)
            digits = set(self._extract_digits(to_exclude))
            self._values = list(vals.difference(digits))

        self._format_string_list()

        return self

    def _format_string_list(self) -> "DataNumComponent":
        """Private method generating a list of formated (BIDS style) strings.

        Raises:
            AttributeError: Raised if the method `_generate_values` wasn't
                            called

        Returns:
            DataNumComponent: The updated instance
        """
        if not getattr(self, "_values", False):
            raise AttributeError(
                "Values were not generated yet. Please \
generate values before calling this method."
            )

        self.formated_strings = []
        for val in self._values:
            str_val = str(val).rjust(self.nb_digit, "0")
            formated_name = self.key + "-" + (self.prefix or "") + str_val

            self.formated_strings.append(formated_name)

        return self

    def generate_formated_list(
        self,
        start: int,
        stop: int,
        component_type: str = 'subject',
        prefix: str | None = None,
        nb_digit=3,
    ) -> "DataNumComponent":
        self.component_type = component_type
        self.prefix = prefix
        self.nb_digit = nb_digit

        self._generate_values(start=start, stop=stop)
        self._format_string_list()


# %%
class DataStrComponent:
    def __init__(self) -> None:
        self.instance_type = "litteral"
        pass
