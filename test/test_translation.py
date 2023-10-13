from arelight.third_party.googletrans import translate_value

x = translate_value("привет", dest="en", src="ru")
print(x)
