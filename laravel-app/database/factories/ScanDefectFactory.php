<?php

namespace Database\Factories;

use App\Models\Scan;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\ScanDefect>
 */
class ScanDefectFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        $scanDefects = [
            'scratch',
            'dent',
            'corrosion',
            'crack',
            'discoloration',
            'missing_part'
        ];

        return [
            //
            'scan_id' => Scan::factory(),
            'label' => $this->faker->randomElement($scanDefects),
            'confidence_score' => $this->faker->randomFloat(2, 0, 1),
            'severity_level' => $this->faker->randomElement(['low', 'medium', 'high']),
            'area_percentage' => $this->faker->randomFloat(2, 0, 100),
            'box_location' => [
                'x' => $this->faker->numberBetween(0, 100),
                'y' => $this->faker->numberBetween(0, 100),
                'width' => $this->faker->numberBetween(1, 100),
                'height' => $this->faker->numberBetween(1, 100),
            ],
            'created_at' => now(),
            'updated_at' => now(),
        ];
    }
}
