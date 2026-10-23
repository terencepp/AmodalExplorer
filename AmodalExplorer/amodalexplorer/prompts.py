"""Local scene prompt formatting adapted from WonderWorld; no API client."""


class PromptBuilder:
    def generate_keywords(self, text):
        import spacy
        doc = spacy.load("en_core_web_sm")(text)

        adj = False
        noun = False
        text = ""
        for token in doc:
            if token.pos_ != "NOUN" and token.pos_ != "ADJ":
                continue

            if token.pos_ == "NOUN":
                if adj:
                    text += (" " + token.text)
                    adj = False
                    noun = True
                else:
                    if noun:
                        text += (", " + token.text)
                    else:
                        text += token.text
                        noun = True
            elif token.pos_ == "ADJ":
                if adj:
                    text += (" " + token.text)
                else:
                    if noun:
                        text += (", " + token.text)
                        noun = False
                        adj = True
                    else:
                        text += token.text
                        adj = True

        return text

    def generate_prompt(self, style, entities, background=None, scene_name=None):
        assert not (background is None and scene_name is None), 'At least one of the background and scene_name should not be None'
        if background is not None:
            if isinstance(background, list):
                background = background[0]

            background = self.generate_keywords(background)
            prompt_text = "Style: " + style + ". Entities: "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                else:
                    prompt_text += (", " + entity)
            prompt_text += (". Background: " + background)
        else:
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            prompt_text = "Style: " + style + ". " + scene_name + " with "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                elif i == len(entities) - 1:
                    prompt_text += (", and " + entity)
                else:
                    prompt_text += (", " + entity)

        return prompt_text
