# You might need to install: pip install piexif
import sys
import os


def inject_xmp(image_path):
    # XMP Metadata template for a 360 Equirectangular Panorama
    # This tells viewers "I am a 360 cylinder/sphere!"
    xmp_data = b"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
    <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.1.0-jc003">
      <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description rdf:about=""
            xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
          <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
          <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
          <GPano:PoseHeadingDegrees>0.0</GPano:PoseHeadingDegrees>
          <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
          <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
          <GPano:FullPanoWidthPixels>{WIDTH}</GPano:FullPanoWidthPixels>
          <GPano:FullPanoHeightPixels>{HEIGHT}</GPano:FullPanoHeightPixels>
          <GPano:CroppedAreaImageWidthPixels>{WIDTH}</GPano:CroppedAreaImageWidthPixels>
          <GPano:CroppedAreaImageHeightPixels>{HEIGHT}</GPano:CroppedAreaImageHeightPixels>
        </rdf:Description>
      </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end="w"?>"""

    try:
        from PIL import Image
    except ImportError:
        print("Please run: pip install pillow")
        return

    print(f"Injecting metadata into {image_path}...")
    img = Image.open(image_path)
    w, h = img.size

    # Fill in the width/height
    xmp_formatted = xmp_data.replace(b"{WIDTH}", str(w).encode()).replace(
        b"{HEIGHT}", str(h).encode()
    )

    # Save with the XMP data
    output_path = image_path.replace(
        ".png", "_360.jpg"
    )  # Convert to JPG for best compatibility
    img = img.convert("RGB")
    img.save(output_path, "JPEG", quality=95, xmp=xmp_formatted)
    print(f"Saved interactive 360 image to {output_path}")


if __name__ == "__main__":
    # Point this to your best panorama
    target_img = "../output/final_panorama_8k.png"
    if os.path.exists(target_img):
        inject_xmp(target_img)
    else:
        print(f"File not found: {target_img}")
