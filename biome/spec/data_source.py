# coding: utf-8

"""
    RecognAI API

    Recognai Platform API specification  # noqa: E501

    OpenAPI spec version: v0.1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from biome.spec.data_source_format import DataSourceFormat  # noqa: F401,E501
from biome.spec.data_source_role import DataSourceRole  # noqa: F401,E501
from biome.spec.data_source_schema import DataSourceSchema  # noqa: F401,E501
from biome.spec.metadata import Metadata  # noqa: F401,E501


class DataSource(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'id': 'str',
        'name': 'str',
        'description': 'str',
        'type': 'str',
        'params': 'dict(str, object)',
        'format': 'DataSourceFormat',
        'schema': 'DataSourceSchema',
        'default_role': 'DataSourceRole',
        'state': 'str',
        'metadata': 'Metadata'
    }

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'description': 'description',
        'type': 'type',
        'params': 'params',
        'format': 'format',
        'schema': 'schema',
        'default_role': 'defaultRole',
        'state': 'state',
        'metadata': 'metadata'
    }

    def __init__(self, id=None, name=None, description=None, type=None, params=None, format=None, schema=None, default_role=None, state=None, metadata=None):  # noqa: E501
        """DataSource - a model defined in Swagger"""  # noqa: E501

        self._id = None
        self._name = None
        self._description = None
        self._type = None
        self._params = None
        self._format = None
        self._schema = None
        self._default_role = None
        self._state = None
        self._metadata = None
        self.discriminator = None

        if id is not None:
            self.id = id
        self.name = name
        if description is not None:
            self.description = description
        self.type = type
        self.params = params
        if format is not None:
            self.format = format
        if schema is not None:
            self.schema = schema
        if default_role is not None:
            self.default_role = default_role
        if state is not None:
            self.state = state
        if metadata is not None:
            self.metadata = metadata

    @property
    def id(self):
        """Gets the id of this DataSource.  # noqa: E501

        The datasource id (autogenerated if not present)  # noqa: E501

        :return: The id of this DataSource.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this DataSource.

        The datasource id (autogenerated if not present)  # noqa: E501

        :param id: The id of this DataSource.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this DataSource.  # noqa: E501

        Unique name of this datasource in its namespace  # noqa: E501

        :return: The name of this DataSource.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this DataSource.

        Unique name of this datasource in its namespace  # noqa: E501

        :param name: The name of this DataSource.  # noqa: E501
        :type: str
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def description(self):
        """Gets the description of this DataSource.  # noqa: E501

        DataSource information summary  # noqa: E501

        :return: The description of this DataSource.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this DataSource.

        DataSource information summary  # noqa: E501

        :param description: The description of this DataSource.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def type(self):
        """Gets the type of this DataSource.  # noqa: E501

        Type of the datasource  # noqa: E501

        :return: The type of this DataSource.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this DataSource.

        Type of the datasource  # noqa: E501

        :param type: The type of this DataSource.  # noqa: E501
        :type: str
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501
        allowed_values = ["FileSystem"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

    @property
    def params(self):
        """Gets the params of this DataSource.  # noqa: E501

        Parameters of the connection to the data. The available parameters depend on the datasource type  # noqa: E501

        :return: The params of this DataSource.  # noqa: E501
        :rtype: dict(str, object)
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the params of this DataSource.

        Parameters of the connection to the data. The available parameters depend on the datasource type  # noqa: E501

        :param params: The params of this DataSource.  # noqa: E501
        :type: dict(str, object)
        """
        if params is None:
            raise ValueError("Invalid value for `params`, must not be `None`")  # noqa: E501

        self._params = params

    @property
    def format(self):
        """Gets the format of this DataSource.  # noqa: E501


        :return: The format of this DataSource.  # noqa: E501
        :rtype: DataSourceFormat
        """
        return self._format

    @format.setter
    def format(self, format):
        """Sets the format of this DataSource.


        :param format: The format of this DataSource.  # noqa: E501
        :type: DataSourceFormat
        """

        self._format = format

    @property
    def schema(self):
        """Gets the schema of this DataSource.  # noqa: E501


        :return: The schema of this DataSource.  # noqa: E501
        :rtype: DataSourceSchema
        """
        return self._schema

    @schema.setter
    def schema(self, schema):
        """Sets the schema of this DataSource.


        :param schema: The schema of this DataSource.  # noqa: E501
        :type: DataSourceSchema
        """

        self._schema = schema

    @property
    def default_role(self):
        """Gets the default_role of this DataSource.  # noqa: E501


        :return: The default_role of this DataSource.  # noqa: E501
        :rtype: DataSourceRole
        """
        return self._default_role

    @default_role.setter
    def default_role(self, default_role):
        """Sets the default_role of this DataSource.


        :param default_role: The default_role of this DataSource.  # noqa: E501
        :type: DataSourceRole
        """

        self._default_role = default_role

    @property
    def state(self):
        """Gets the state of this DataSource.  # noqa: E501

        Data source state * UNKNOWN  The data source state is unspecified. * READY The data source is ready for be used inside an ecosystem. * CREATING  The data source is being created. 'Being created' means that some needed information must be provided (formats, schemas...). * INDEXING  The data source is being indexing. * DELETING  The data source is being deleted. * FAILED  The data source failed to be indexed.   # noqa: E501

        :return: The state of this DataSource.  # noqa: E501
        :rtype: str
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this DataSource.

        Data source state * UNKNOWN  The data source state is unspecified. * READY The data source is ready for be used inside an ecosystem. * CREATING  The data source is being created. 'Being created' means that some needed information must be provided (formats, schemas...). * INDEXING  The data source is being indexing. * DELETING  The data source is being deleted. * FAILED  The data source failed to be indexed.   # noqa: E501

        :param state: The state of this DataSource.  # noqa: E501
        :type: str
        """
        allowed_values = ["UNKNOWN", "READY", "CREATING", "INDEXING", "DELETING", "FAILED"]  # noqa: E501
        if state not in allowed_values:
            raise ValueError(
                "Invalid value for `state` ({0}), must be one of {1}"  # noqa: E501
                .format(state, allowed_values)
            )

        self._state = state

    @property
    def metadata(self):
        """Gets the metadata of this DataSource.  # noqa: E501


        :return: The metadata of this DataSource.  # noqa: E501
        :rtype: Metadata
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Sets the metadata of this DataSource.


        :param metadata: The metadata of this DataSource.  # noqa: E501
        :type: Metadata
        """

        self._metadata = metadata

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, DataSource):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
