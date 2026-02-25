from .models import ActivityTemplate, EntityType, ImplicitEntity, SpatialHint

TEMPLATES = [
    ActivityTemplate(
        name="barbecue",
        trigger_verbs={"barbecue", "grill", "bbq", "cook"},
        trigger_nouns={"barbecue", "grill", "bbq"},
        expected_person_count=2,
        implicit_entities=[
            ImplicitEntity(
                name="barbecue grill", entity_type=EntityType.OBJECT,
                role="equipment",
                default_dimensions={"height": 1.0, "width": 0.6, "length": 0.6},
                default_mass=30.0,
                mesh_prompt="a charcoal barbecue grill with round lid, metal, outdoor",
            ),
            ImplicitEntity(
                name="folding table", entity_type=EntityType.OBJECT,
                role="furniture", required=False,
                default_dimensions={"height": 0.75, "width": 0.60, "length": 1.20},
                default_mass=8.0,
                mesh_prompt="a folding outdoor table, plastic, white",
            ),
            ImplicitEntity(
                name="lawn chair", entity_type=EntityType.OBJECT,
                role="furniture", required=False,
                default_dimensions={"height": 0.90, "width": 0.55, "length": 0.55},
                default_mass=4.0,
                mesh_prompt="a folding lawn chair, fabric and metal",
            ),
        ],
        spatial_hints=[
            SpatialHint("person", "grill", "in_front_of", 0.8),
            SpatialHint("table", "grill", "beside", 1.5),
        ],
        default_setting="backyard",
    ),

    ActivityTemplate(
        name="romantic_dance",
        trigger_verbs={"dance", "waltz", "tango"},
        trigger_nouns={"dance"},
        expected_person_count=2,
        spatial_hints=[SpatialHint("person", "person", "facing", 0.5)],
        default_setting="ballroom",
    ),

    ActivityTemplate(
        name="car_racing",
        trigger_verbs={"race", "speed", "drive", "accelerate"},
        trigger_nouns={"race", "racing"},
        implicit_entities=[
            ImplicitEntity(
                name="road", entity_type=EntityType.ENVIRONMENT,
                role="surface",
                mesh_prompt="a straight asphalt road with lane markings",
            ),
        ],
        spatial_hints=[
            SpatialHint("vehicle", "vehicle", "beside", 3.0),
            SpatialHint("vehicle", "road", "on_top_of", 0.0),
        ],
        default_setting="road",
    ),

    ActivityTemplate(
        name="football_game",
        trigger_verbs={"kick", "pass", "score", "play"},
        trigger_nouns={"football", "soccer", "ball"},
        expected_person_count=2,
        implicit_entities=[
            ImplicitEntity(
                name="football", entity_type=EntityType.OBJECT,
                role="ball",
                default_dimensions={"diameter": 0.22},
                default_mass=0.43,
                mesh_prompt="a classic black and white soccer ball",
            ),
            ImplicitEntity(
                name="goal post", entity_type=EntityType.OBJECT,
                role="goal", required=False,
                default_dimensions={"height": 2.44, "width": 7.32, "length": 0.12},
                default_mass=80.0,
                mesh_prompt="a white football goal post with net",
            ),
        ],
        spatial_hints=[SpatialHint("person", "football", "in_front_of", 0.5)],
        default_setting="grass_field",
    ),

    ActivityTemplate(
        name="throwing",
        trigger_verbs={"throw", "toss", "launch", "hurl"},
        expected_person_count=1,
        default_setting="outdoor",
    ),

    ActivityTemplate(
        name="dining",
        trigger_verbs={"eat", "dine", "sit"},
        trigger_nouns={"dinner", "lunch", "meal", "restaurant", "table"},
        expected_person_count=2,
        implicit_entities=[
            ImplicitEntity(
                name="dining table", entity_type=EntityType.OBJECT,
                role="furniture",
                default_dimensions={"height": 0.75, "width": 0.90, "length": 1.50},
                default_mass=25.0,
                mesh_prompt="a rectangular wooden dining table",
            ),
            ImplicitEntity(
                name="chair", entity_type=EntityType.OBJECT,
                role="furniture",
                default_dimensions={"height": 0.90, "width": 0.45, "length": 0.45},
                default_mass=5.0,
                mesh_prompt="a wooden dining chair with backrest",
            ),
        ],
        spatial_hints=[
            SpatialHint("person", "table", "beside", 0.4),
            SpatialHint("chair", "table", "beside", 0.0),
        ],
        default_setting="dining_room",
    ),

    ActivityTemplate(
        name="walking",
        trigger_verbs={"walk", "stroll", "jog", "run", "hike"},
        default_setting="park",
    ),
]
