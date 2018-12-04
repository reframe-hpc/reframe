import collections
import reframe.utility.sanity


class CheckList(collections.UserList):

    def __init__(self, initial_list):
        self.data = initial_list #TODO: use just one list
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

    def filter_by_tags(self, tags):# TOD: why do we need a set here?
        self.filter(lambda c:
            c if tags.issubset(c.tags) else None)

    def filter(self, function):
        self.filtered_data = filter(function, self.filtered_data) 

