import collections
import reframe.utility.sanity


class CheckList(collections.UserList):

    def __init__(self, initial_list):
        self.data = initial_list #TODO: we only need this if we implement a "reset" method
        self.filtered_data = initial_list

    def filter_by_names(self, names, exclude=False):
        if exclude == False:
            self.filter(lambda c:
                c if c.name in names else None)
        elif exclude == True:
            self.filter(lambda c:
                c if c.name not in names else None)
        else:
            # This should not happen
            # TODO: should I use an exception here?
            printer.error("filter_by_names errror, exclude parameter is a boolean")

    def filter_by_tags(self, tags):# TODO: why do we need a set here?
        self.filter(lambda c:
            c if tags.issubset(c.tags) else None)

    def filter(self, function):
        self.filtered_data = filter(function, self.filtered_data) 
        self.filtered_data = [c for c in self.filtered_data]

    def get_check_list(self) -> list:
        return self.filtered_data
