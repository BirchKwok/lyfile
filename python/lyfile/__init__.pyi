from typing import Union
from pathlib import Path

import pandas as pd
import pyarrow as pa


class LyFile:
    def __init__(self, filepath: Union[str, Path]):
        """
        Initializes a new LyFile object.

        Args:
            filepath (str): The path of the file to read or write.

        Returns:
            LyFile: A new instance of LyFile.

        Examples:
            >>> from lyfile import LyFile
            >>> lyfile = LyFile("example.ly")
        """
        ...

    def write(self, data: Union[pd.DataFrame, dict, pa.Table]):
        """
        Writes data to the custom file format.

        Args:
            data (Union[pandas.DataFrame, dict, pyarrow.Table]):
                The input data to be written.
                Supported types include:
                - Pandas DataFrame
                - Python dictionary
                - PyArrow Table

        Raises:
            ValueError: If the input data type is not supported or is empty.
            IOError: If an error occurs while writing to the file.
            ArrowError: If there is an error with Arrow serialization.
            SerializationError: If an error occurs during metadata serialization.

        Examples:
            >>> import pandas as pd
            >>> from lyfile import LyFile
            >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            >>> lyfile = LyFile("example.ly")
            >>> lyfile.write(df)
        """
        ...

    def read(self, columns: Union[str, list[str]]=None):
        """
        Reads data from the custom file format.

        Args:
            columns (Optional[Union[str, List[str]]]):
                The names of columns to read. If None, all columns are read.

        Returns:
            pyarrow.Table: The data read from the file.

        Raises:
            ValueError: If the file does not exist or the column names are invalid.
            IOError: If an error occurs while reading the file.

        Examples:
            >>> from lyfile import LyFile
            >>> lyfile = LyFile("example.ly")
            >>> data = lyfile.read()
            >>> print(data)
        """
        ...

    @property
    def shape(self):
        """
        Returns the shape of the data in the file.
        """
        ...

    def append(self, data: Union[pd.DataFrame, dict, pa.Table]):
        """Appends data to the existing file.

        Args:
            data (Union[pandas.DataFrame, dict, pyarrow.Table]):
                The input data to be appended.
                Supported types include:
                - Pandas DataFrame
                - Python dictionary
                - PyArrow Table

        Raises:
            ValueError: If the input data type is not supported or if schemas do not match.
            IOError: If an error occurs while writing to the file.
            ArrowError: If there is an error with Arrow serialization.
            SerializationError: If an error occurs during metadata serialization.

        Examples:
            >>> import pandas as pd
            >>> from lyfile import LyFile
            >>> df = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
            >>> lyfile = LyFile("example.ly")
            >>> lyfile.append(df)
        """
        ...