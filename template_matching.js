const cv = require("opencv4nodejs-prebuilt-install");
const fs = require("fs");
const path = require("path");

/**
 * @param output Raw output of YOLOv8 network
 * @param img_width Width of original image
 * @param img_height Height of original image
 * @returns Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
 */
async function template_matching(originalMat, waldoMat) {
  let originalMat_gray = originalMat.cvtColor(cv.COLOR_BGR2GRAY);
  let waldoMat_gray = waldoMat.cvtColor(cv.COLOR_BGR2GRAY);

  // Match template (the brightest locations indicate the highest match)
  const matched = originalMat_gray.matchTemplate(waldoMat_gray, cv.TM_SQDIFF);

  // Use minMaxLoc to locate the highest value (or lower, depending of the type of matching method)
  const minMax = matched.minMaxLoc();
  const {
    minLoc: { x, y },
  } = minMax;

  // originalMat.drawRectangle(
  //   new cv.Rect(x, y, waldoMat.cols, waldoMat.rows),
  //   new cv.Vec(0, 255, 0),
  //   2,
  //   cv.LINE_8
  // );
  // cv.imshow('We\'ve found Waldo!', originalMat);
  // cv.waitKey();

  return [x, y, x + waldoMat.cols, y + waldoMat.rows, 0, 1.];
}

/**
 * Function calculates "Intersection-over-union" coefficient for specified two boxes
 * https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
 * @param box1 First box in format: [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format: [x1,y1,x2,y2,object_class,probability]
 * @returns Intersection over union ratio as a float number
 */
function iou(box1, box2) {
  return intersection(box1, box2) / union(box1, box2);
}

/**
 * Function calculates union area of two boxes.
 *     :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
 *     :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
 *     :return: Area of the boxes union as a float number
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of the boxes union as a float number
 */
function union(box1, box2) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
  const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
  const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
  return box1_area + box2_area - intersection(box1, box2);
}

/**
 * Function calculates intersection area of two boxes
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of intersection of the boxes as a float number
 */
function intersection(box1, box2) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
  const x1 = Math.max(box1_x1, box2_x1);
  const y1 = Math.max(box1_y1, box2_y1);
  const x2 = Math.min(box1_x2, box2_x2);
  const y2 = Math.min(box1_y2, box2_y2);
  return (x2 - x1) * (y2 - y1);
}

/**
 * Function that calculates the average precision of a class at a given IoU threshold
 * @param {Array} predictions Array of detected objects
 * @param {Array} gt Array of ground truth bounding boxes
 * @param {Number} class_index Index of the class to calculate the average precision for
 * @param {Number} iou_threshold IoU threshold at which to calculate the average precision
 * @returns {Number} Average precision of the class at the given IoU threshold
 */
function average_precision_recall(predictions, gt, class_index, iou_threshold) {
  const class_predictions = predictions;
  const class_gt = gt;

  let tp_p = Array(class_predictions.length).fill(false);
  let tp_r = Array(class_gt.length).fill(false);
  class_predictions.map((predBox, pred_id) => {
    class_gt.map((gtBox, gt_id) => {
      if (iou(predBox, gtBox) >= iou_threshold) {
        tp_p[pred_id] = true;
        tp_r[gt_id] = true;
      }
    });
  });

  const precision =
    class_predictions.length > 0
      ? tp_p.filter((v) => v === true).length / class_predictions.length
      : 1;
  const recall =
    class_gt.length > 0
      ? tp_r.filter((v) => v === true).length / class_gt.length
      : 1;
  return [precision, recall];
}

/**
 * Function that calculates the mAP50-95 of the model
 * @param {String} imagePath Path to the image to be detected
 * @returns {Promise} Promise that resolves to the mAP50-95 of the model
 */
async function calculate_map(imagePath, labelPath) {
  console.log(imagePath);
  let originalMat;
  try {
    originalMat = await cv.imreadAsync(imagePath);
  } catch (err) {
    console.error(`Failed to read image: ${err}`);
    return [0, 0];
  }
  const gt = get_ground_truth(
    labelPath,
    originalMat.cols,
    originalMat.rows
  );

  const boxes = await Promise.all(gt.map(async (gold) => {
    const subMat = originalMat.getRegion(new cv.Rect(gold[0], gold[1], gold[2] - gold[0], gold[3] - gold[1]));
    return await template_matching(originalMat, subMat);
  }));
  const predictions = boxes;
  // console.log([gt, predictions]);

  const class_aps = [];
  const class_res = [];
  for (let j = 0.5; j <= 0.95; j += 0.05) {
    const [ap, re] = average_precision_recall(predictions, gt, 0, j);
    class_aps.push(ap);
    class_res.push(re);
  }
  const ap = class_aps.reduce((acc, val) => acc + val, 0) / class_aps.length;
  const re = class_res.reduce((acc, val) => acc + val, 0) / class_res.length;
  return [ap, re];
}

/**
 * Function to get ground truth bounding boxes from an image
 * @param {String} imagePath Path to the image to be detected
 * @returns {Promise} Promise that resolves to the ground truth bounding boxes
 */
function get_ground_truth(labelPath, img_width, img_height) {
  const lines = fs.readFileSync(labelPath, "utf8").split("\n");
  return lines.map((line) => {
    if (!line.trim()) return [];
    const [label, x, y, w, h] = line.split(" ");
    x1 = (parseFloat(x) - parseFloat(w) / 2) * img_width;
    y1 = (parseFloat(y) - parseFloat(h) / 2) * img_height;
    x2 = (parseFloat(x) + parseFloat(w) / 2) * img_width;
    y2 = (parseFloat(y) + parseFloat(h) / 2) * img_height;
    return [x1, y1, x2, y2, label, 1];
  });
}

(async () => {
  const startTime = Date.now();
  const folderPath = "./data/images";
  const files = fs.readdirSync(folderPath);
  let [scores_p, scores_r] = [[], []];
  let cnt = 0;
  for (const file of files) {
    cnt += 1;
    const filePath = path.join(folderPath, file);
    const labelPath = path.join("./data/labels", file.slice(0, -4) + ".txt");
    const startTime = Date.now();
    let [precision, recall] = await calculate_map(filePath, labelPath);
    scores_p.push(precision);
    scores_r.push(recall);
    const endTime = Date.now();
    console.log(`${cnt}: ${precision}, ${recall}, time: ${(endTime - startTime) / 1000}s`);
  }
  const endTime = Date.now();
  console.log(
    `length: ${scores_p.length}, mAP: ${
      scores_p.reduce((acc, val) => acc + val, 0) / scores_p.length
    }, mAR: ${scores_r.reduce((acc, val) => acc + val, 0) / scores_r.length}, tbc time: ${(endTime - startTime) / scores_p.length / 1000}s`
  );
})();

// (async () => {
//   const folderPath = "./data/images";
//   let st = "Table--194-_jpg.rf.575d11723e72356aa9977dd4e50b55ec"
//   const filePath = path.join(folderPath, st+".jpg");
//   const labelPath = path.join("./data/labels", st+".txt");
//   console.log(await calculate_map(filePath, labelPath))
// })();