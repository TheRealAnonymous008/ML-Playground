
from random import choices

def _drop(x):
    return ""

def _retain(x):
    return x 

def _choose(x :  str | dict[str, float]):
    if type(x) is str: 
        return x 
    
    keys = list(x.keys())
    weights = list(x.values())
    
    # Use random.choices to select a key based on weights
    chosen_key = choices(keys, weights=weights, k=1)[0]
    
    return chosen_key

def _translate(c) :
    match c:
        # Syllabication
        case ".": return _drop(c)
        # Stress Mark
        case "ˌ": return _drop(c)         
        case "-": return _drop(c)
        case "ˈ": return _drop(c)
        # Tone Marks
        case "¹": return _drop(c)
        case "²": return _drop(c)
        case "³": return _drop(c)
        case "⁴": return _drop(c) 
        case "⁵": return _drop(c)

        case "˥": return _drop(c)
        case "˦": return _drop(c)
        case "˧": return _drop(c)
        case "˨": return _drop(c)
        case "˩": return _drop(c)

        # Vowel Sounds

        # Front
        case "i": return _retain("ee")
        case "ĩ": return _translate("i")
        case "y": return _retain("ü")
        case "ɪ": return _retain("i")
        case "ʏ": return _retain("w")
        case "e": return _retain("é")
        case "ẽ": return _translate("e")
        case "ø": return _retain("ø")

        case "ɛ": return _retain("e")
        case "œ": return _retain("ö")
        case "æ": return _retain("æ")
        case "a": return _retain("a")
        case "ã": return _translate("a")
        case "ɶ": return _retain("ē")

        # Central
        case "ɨ": return _retain("y")
        case "ʉ": return _retain("uu")
        case "ɘ": return _retain("eu")
        case "ɵ": return _retain("oe")
        case "ɜ": return _retain("ë")
        case "ɞ": return _retain("ô")
        case "ɐ": return _retain("ā")
        case "ä": return _retain("ä")

        # Mid Sounds
        case "e̞" | "ø̞" | "ə" | "ɤ̞" | "o̞": 
            return _choose({"a" : 1, "e" : 1, "i" : 1, "o": 1, "u" : 1})

        # Back
        case "ɯ": return _retain("ao")
        case "u": return _retain("oo")
        case "ʊ": return _retain("u")
        case "ũ": return _translate("u")
        case "ɤ": return _retain("õ")
        case "o": return _retain("o")
        case "ʌ": return _retain("à")
        case "ɔ": return _retain("ó")
        case "ɑ": return _retain("aw")
        case "ɒ": return _retain("ă")
        
        #  Misc
        case "ɝ" | "ɚ": return _retain("r")
        case "ɥ": return _choose({"u" : 1, "w" : 1})

        # Consonants
        case "m̥" | "m": return _retain("m")
        case "ɱ̊" | "ɱ": return _retain("m")
        case "n̼"      : return _retain("n")
        case "n": return _retain("n")
        case "ɳ": return _retain("n")
        case "ɲ": return _retain("ñ")
        case "ŋ": return _retain("ng")
        case "ɴ": return _retain("ng")

        case "p": return _retain("p")
        case "b": return _retain("b")
        case "ɓ": return _retain("b")
        case "t": return _retain("t")
        case "d": return _retain("d")
        case "ʈ": return _retain("t")
        case "ɗ": return _retain("d")
        case "ɖ": return _retain("d")
        case "c": return _retain("ċ")
        case "ɟ": return _retain("ġ")
        case "k": return _retain("k")
        case "ɡ": return _retain("g")
        case "q": return _retain("c")
        case "ɢ": return _choose({"c" : 1, "g" : 1})
        case "ʡ": return _retain("'")
        case "ʔ": return _retain("'")
        
        case "s": return _retain("s")
        case "z": return _retain("z") 
        case "ʃ": return _retain("sh")
        case "ʒ": return _retain("zh")
        case "ʂ": return _choose({"š" : 1, "ш" : 1})
        case "ʐ": return _choose({"ž" : 1, "ж" : 1})
        case "ɕ": return _retain("kj")
        case "ʑ": return _choose({"sj" : 1, "жь" : 1})
        case "ɧ": return _retain("sj")

        case "ɸ": return _retain("fh")
        case "β": return _retain("vh")
        case "f": return _retain("f")
        case "v": return _retain("v")
        case "θ": return _retain("th")
        case "ð": return _retain("dh")
        case "ç": return _retain("ç") 
        case "ʝ": return _retain("j")
        case "x": return _retain("ch")
        case "ɣ": return _retain("gh")
        case "χ": return _retain("ckh")
        case "ʁ": return _retain("r")
        case "ħ": return _retain("kh")
        case "ʕ": return _retain("wr")
        case "h": return _retain("h")
        case "ɦ": return _retain("h'")

        case "ʋ": return _choose({"vw" : 1, "w" : 1})
        case "ɹ": return _retain("r")
        case "ɻ": return _retain("rh")
        case "j": return _choose({"y" : 1, "j" : 1})
        case "ɰ": return _retain("gw")

        case "ⱱ": return _retain("bb")
        case "ɾ": return _retain("r")
        case "ɽ": return _retain("r")
        case "ɢ̆": return _retain("g'")
        case "ʡ̆": return _retain("k'")

        case "ʙ": return _retain("pb")
        case "r": return _retain("rr")
        case "ʀ": return _retain("rr")
        case "ʜ": return _retain("hk")
        case "ʢ": return _retain("h'r")

        case "ɬ": return _choose({"lt" : 1, "ḻ" : 1})
        case "ɮ": return _retain("lzh")
        case "ꞎ": return _choose({"lt" : 1, "ḻ" : 1})
        case "𝼄": return _retain("ly")

        case "l": return _retain("l")
        case "ɭ": return _retain("ll")
        case "ʎ": return _retain("ly")
        case "ʟ": return _retain("lh")

        case "ɺ": return _retain("l'")
        case "ɫ": return _retain("l")

        case "w": return _retain("w")

        case "ˀ": return _retain("'")

        # Misc Symbols
        case "(" | ")": return _drop(c)   # Drop parentheses
        case "ː" | ":": return _drop(c)         # Drop prolong symbol

        # Diacritics
        case "̥" | "̊̊": return _retain("h")       # Voiceless
        case "ʱ" | "ʰ": return _drop(c)         # Aspirated
        case "̌̌": return _drop(c)                # Voiced    
        case "⁾" | "⁽": return _drop(c)         # More / Less Rounded
        case "̟": return _drop(c)                # Advanced
        case "̠": return _drop(c)                # Retracted
        case "̈̈": return _drop(c)                # Centralized
        case "̽": return _drop(c)                # Mid centralized 
        case "̩": return _drop(c)                # Syllabic
        case "̯": return _drop(c)                # Non syllabic
        case "˞": return _drop(c)                # Rhoticity
        case "̤": return _drop(c)                # Breathy Voiced
        case "̰": return _drop(c)                # Creaky Voiced
        case "̼": return _drop(c)                # Lingolabial
        case "ʷ": return _drop(c)               # Labialized
        case "ᵝ": return _drop(c)
        case "ʲ": return _drop(c)               # Palatized
        case "ˠ":  return _drop(c)              # Velarized
        case "ˤ": return _drop(c)               # Pharyngeal
        case "̴": return _drop(c)                # Velarized / Pharyngealized
        case "̝": return _drop(c)                # Raised
        case "̞": return _drop(c)                # Lowered
        case "̘": return _drop(c)                # Advanced Tongue root
        case "̙": return _drop(c)                # Retracted Tongue root
        case "̪": return _drop(c)                # Dental
        case "̺" : return _drop(c)               # Apical
        case "̻": return _drop(c)                # Laminal
        case "̃": return _drop(c)                # Nasalized
        case "ⁿ": return _drop(c)               # Nasal Release
        case "ˡ": return _drop(c)               # Lateral Release
        case "̚": return _drop(c)                # No audible release
        case "~": return _drop(c)
        case "ˣ": return _drop(c)

        case "͜" | "͡" | "‿": return _drop(c)
        case " ": return _drop(c)
        case "[" | "]": return _drop(c)
        case "̯": return _drop(c)
        case "͈ ": return _drop(c)
        case "⁻": return _drop(c)
        case " ͈": return _drop(c)
        case "ˑ": return _drop(c)
        case " ̹": return _drop(c)
        case _: 
            # print(f"Encountered unknown symbol : {c}")
            return _drop(c)

def romanize(x : str):
    y = ""

    for c in x: 
        y += _translate(c)
    return y