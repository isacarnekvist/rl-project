import abc


class PolicyABC(abc.ABC):

    @abc.abstractmethod
    def act(self, state):
        """
        All actors should override this
        :param state: list or 1-d numpy array
        :return: list or 1-d numpy array with action
        """
        pass


if __name__ == '__main__':
    actor = PolicyABC()