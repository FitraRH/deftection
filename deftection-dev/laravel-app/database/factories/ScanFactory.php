<?php

namespace Database\Factories;

use App\Models\User;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\Scan>
 */
class ScanFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $format = $this->faker->randomElement(['jpg', 'png', 'jpeg']);

        return [
            'user_id' => User::factory(),
            'filename' => $this->faker->word . '.' . $format,
            'original_path' => $this->faker->imageUrl(640, 480, null, true, null, false, $format),
            'annotated_path' => $this->faker->imageUrl(640, 480, null, true, null, false, $format),
            'is_defect' => $this->faker->boolean,
            'anomaly_score' => $this->faker->randomFloat(5, 0, 1),
            'anomaly_confidence_level' => $this->faker->randomElement(['low', 'medium', 'high']),
            'anomaly_inference_time_ms' => $this->faker->randomFloat(3, 0, 1000),
            'classification_inference_time_ms' => $this->faker->randomFloat(3, 0, 1000),
            'preprocessing_time_ms' => $this->faker->randomFloat(3, 0, 1000),
            'postprocessing_time_ms' => $this->faker->randomFloat(3, 0, 1000),
            'anomaly_threshold' => $this->faker->randomFloat(5, 0, 1),
            'created_at' => now(),
            'updated_at' => now(),
        ];
    }
}
