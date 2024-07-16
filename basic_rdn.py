import omni.replicator.core as rep

with rep.new_layer():
    camera = rep.create.camera(position=(-4, 9, 1), look_at=(-5,30, 1))
    #camera = rep.create.camera(position=(-4, -9, 1), look_at=(-5,30, 1))
    #camera = rep.create.camera(position=(-4, 9, 3), look_at=(10,0, 2))
    #camera = rep.create.camera(position=(7, -8, 5), look_at=(-6,10, 3))
    #camera = rep.create.camera(position=(7, -8, 5), look_at=(0,30, 1))
    #camera = rep.create.camera(position=(-3, 9, 5), look_at=(0,0, 1))
    
    render_product = rep.create.render_product(camera, (1024, 1024))
    with rep.trigger.on_frame(num_frames=10):
        boxes = rep.get.prims(semantics=[('class', 'chair')])
        with boxes:
            rep.modify.pose(
                position=rep.distribution.uniform((-3, -14, .5), (-5, -16, .5)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 90)),
                scale=rep.distribution.uniform(1, 1.5))
    
    # Attach 2D tight bounding box annotator
    bbox_2d_tight = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    #sd.SyntheticData.Get().set_instance_mapping_semantic_filter("class:chair")
    bbox_2d_tight.filter.prims = boxes
    #bbox_2d_tight.filter.sem_type = "class"
    #bbox_2d_tight.filter.sem_data = "chair"
    bbox_2d_tight.attach(render_product)
    
    
    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize( output_dir="_output", rgb=True,   bounding_box_2d_tight=True)
    writer.attach([render_product])
    rep.orchestrator.preview()