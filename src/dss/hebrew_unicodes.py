class HebrewUnicodes:
    """
    This class contains the unicodes for Hebrew characters.
    """
    CHARS = {
        'ALEF': '\u05D0',
        'AYIN': '\u05e2',
        'BET': '\u05D1',
        'DALET': '\u05D3',
        'GIMEL': '\u05D2',
        'HE': '\u05D4',
        'HET': '\u05D7',
        'KAF': '\u05DB',
        'KAF_FINAL': '\u05DA',
        'LAMED': '\u05DC',
        'MEM': '\u05DD',
        'MEM_MEDIAL': '\u05DE',
        'NUN_FINAL': '\u05DF',
        'NUN_MEDIAL': '\u05E0',
        'PE': '\u05E4',
        'PE_FINAL': '\u05E3',
        'QOF': '\u05E7',
        'RESH': '\u05E8',
        'SAMEKH': '\u05E1',
        'SHIN': '\u05E9',
        'TAW': '\u05EA',
        'TET': '\u05D8',
        'TSADI_FINAL': '\u05E5',
        'TSADI_MEDIAL': '\u05E6',
        'WAW': '\u05D5',
        'YOD': '\u05D9',
        'ZAYIN': '\u05D6',
    }

    @classmethod
    def characters(cls):
        """
        Returns the list of Hebrew characters.
        :return: The list of Hebrew characters.
        """
        return list(cls.CHARS.values())

    @classmethod
    def name_to_unicode(cls, name: str):
        """
        Returns the unicode for the given Hebrew character name.
        :param name: The name of the Hebrew character.
        :return: The unicode for the given Hebrew character name.
        """
        return cls.CHARS[name.replace('-', '_').upper()]
