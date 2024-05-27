
class CellAdder:
    indices = ["ACI", "ADI", "AEI", "BIO"]
    headings = ["Big Vicky", "Carara"]
    flts = ["extracted", "filtered"]
    bands = ["broadband", "fish", "shrimp"]

    def __init__(self, add_cell) -> None:
        self.add_cell = add_cell

    @classmethod
    def generate_data_cell(cls, heading, band, index, factors, cross_effect, marine, flt=None):
        if flt:
            name = f"{heading}_{flt}_{band}_{index}"
        else:
            name = f"{heading}_{band}_{index}"

        if band == "broadband":
            group = f"{band}_{index}_group"
        else:
            group = f"{band}_{flt}_{index}_group"

        text =  [f"{name}_fig, {name}_ratio_summary = experiment({heading}_df,", 
                 f"{group},",  
                 f"{heading}_toolbox,", 
                 "r_link,", 
                 f"marine={marine},", 
                 f"factors={factors},",
                 f"cross_effect='{cross_effect}')"]

        for i, txt in enumerate(text[1:]):
            text[i+1] = f"{' ' * 10}{txt}"

        return text
    
    @classmethod
    def generate_print_cell(cls, heading, band, index, flt=None):
        if flt:
            return f"print({heading}_{flt}_{band}_{index}_ratio_summary)"

        return f"print({heading}_{band}_{index}_ratio_summary)"

    
    @classmethod
    def generate_fig_cell(cls, heading, band, index, flt=None):
        if flt:
            return f"{heading}_{flt}_{band}_{index}_fig"

        return f"{heading}_{band}_{index}_fig"
    
    @classmethod
    def generate_md_cell(cls, text, level):
        return f"{'#'*level} {text}"
    
    @classmethod
    def generate_index_definitions(cls):
        string = ""
        for band in cls.bands:
            for flt in cls.flts:
                if band == 'broadband' and flt == 'filtered':
                    continue

                for index in cls.indices:
                    if band == "broadband":
                        variable = f"{band}_{index}_group"
                    else:
                        variable = f"{band}_{flt}_{index}_group"

                    string += f"{variable} = ('{index}', {True if flt=='filtered' else False}, '{band}')\n"

        return string

    def add_vicky_cells(self):
        heading = "Big Vicky"
        small_heading = 'vicky'
        factors = ["Hour", "Site"]
        cross_effect = "Hour"
        for index in self.indices:
            for flt in self.flts:
                for band in self.bands:
                    if band == "broadband" and flt == "filtered": continue
                    ttl = f"{heading} - {'' if band == 'broadband' else flt} {band} {index}".replace('  ', ' ')
                    self.add_cell(self.generate_md_cell(ttl, 3), 'markdown')
                    self.add_cell(self.generate_data_cell(small_heading, band, index, factors, cross_effect, True, flt), 'code')
                    self.add_cell(self.generate_print_cell(small_heading, band, index, flt), 'code')
                    self.add_cell(self.generate_fig_cell(small_heading, band, index, flt), 'code')

    def add_carara_cells(self):
        heading = "Carara"
        small_heading = 'carara'
        factors = ["Site"]
        cross_effect = "Site"
        flt = "extracted"
        band = "broadband"
        for index in self.indices:
            ttl = f"{heading} - {'' if band == 'broadband' else flt} {band} {index}".replace('  ', ' ')
            self.add_cell(self.generate_md_cell(ttl, 3), 'markdown')
            self.add_cell(self.generate_data_cell(small_heading, band, index, factors, cross_effect, False), type='code')
            self.add_cell(self.generate_print_cell(small_heading, band, index), 'code')
            self.add_cell(self.generate_fig_cell(small_heading, band, index), 'code')

if __name__ == "__main__":
    from functools import partial
    import json

    def format_cell_text(text):
        return text.replace('\n','\\n').replace("\"", "\\\"").replace("'", "\'")
    def add_cell(l, text, type):
        if isinstance(text, list):
            for i, txt in enumerate(text):
                text[i] = format_cell_text(txt)
                text[i] = f"{text[i]}\n"
        else:
            text = format_cell_text(text)
            text = [f"{text}\n"]

        cell_data = {
        "cell_type": type,
        "metadata": {},
        "source": text
        }

        l.append(cell_data)

    adder = CellAdder(add_cell)
    for i, func in enumerate([adder.add_vicky_cells, adder.add_carara_cells]):
        cell_list = []
        add_cell_to_list = partial(add_cell, cell_list)
        adder.add_cell = add_cell_to_list
        func()

        with open(f"out_{i}.json", 'w') as f:
            json.dump(cell_list, f, indent=2)

    print(adder.generate_index_definitions())