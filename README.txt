# LABELING: Scores for alle plader

# IMAGE DETECTION PHASE:
    # 1 Opdel spillepladerne i 5x5 felter
    # 2 Label terræntyperne enkeltvis (implementer ML til dette, måske CVAT)
    # 3 Udskær krone (CVAT)
    # 3 Template matching på kongekroner for at se om der er krone på feltet
    # 4 BLOB detection (grass-fire)
    # 5 Tæl felter i BLOBS, tæl antal af kroner, udregn hvor mange point den blob udgør
    # 6 Tag sum af blob point 

